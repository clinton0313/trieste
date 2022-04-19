# Copyright 2021 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Dict, Optional, Sequence

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...types import TensorType
from ..interfaces import (
    EnsembleModel,
    EvidentialPriorModel,
    HasTrajectorySampler,
    EvidentialPriorModel,
    TrainableProbabilisticModel,
    TrajectorySampler,
)
from ..optimizer import KerasOptimizer
from .architectures import KerasEnsemble, DeepEvidentialNetwork
from .interface import KerasPredictor
from .sampler import EnsembleTrajectorySampler, DeepEvidentialTrajectorySampler
from .utils import (
    DeepEvidentialCallback,
    negative_log_likelihood,
    normal_inverse_gamma_negative_log_likelihood,
    normal_inverse_gamma_regularizer,
    sample_with_replacement,
)


class DeepEnsemble(
    KerasPredictor, TrainableProbabilisticModel, EnsembleModel, HasTrajectorySampler
):
    """
    A :class:`~trieste.model.TrainableProbabilisticModel` wrapper for deep ensembles built using
    Keras.

    Deep ensembles are ensembles of deep neural networks that have been found to have good
    representation of uncertainty in practice (<cite data-cite="lakshminarayanan2017simple"/>).
    This makes them a potentially attractive model for Bayesian optimization for use-cases with
    large number of observations, non-stationary objective functions and need for fast predictions,
    in which standard Gaussian process models are likely to struggle. The model consists of simple
    fully connected multilayer probabilistic networks as base learners, with Gaussian distribution
    as a final layer, using the negative log-likelihood loss for training the networks. The
    model relies on differences in random initialization of weights for generating diversity among
    base learners.

    The original formulation of the model does not include boostrapping of the data. The authors
    found that it does not improve performance the model. We include bootstrapping as an option
    as later work that more precisely measured uncertainty quantification found that boostrapping
    does help with uncertainty representation (see <cite data-cite="osband2021epistemic"/>).

    We provide classes for constructing ensembles using Keras
    (:class:`~trieste.models.keras.KerasEnsemble`) in the `architectures` package that should be
    used with the :class:`~trieste.models.keras.DeepEnsemble` wrapper. There we also provide a
    :class:`~trieste.models.keras.GaussianNetwork` base learner following the original
    formulation in <cite data-cite="lakshminarayanan2017simple"/>, but any user-specified network
    can be supplied, as long as it has a Gaussian distribution as a final layer and follows the
    :class:`~trieste.models.keras.KerasEnsembleNetwork` interface.

    Note that currently we do not support setting up the model with dictionary configs and saving
    the model during Bayesian optimization loop (``track_state`` argument in
    :meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method should be set to `False`).
    """

    def __init__(
        self,
        model: KerasEnsemble,
        optimizer: Optional[KerasOptimizer] = None,
        bootstrap: bool = False,
    ) -> None:
        """
        :param model: A Keras ensemble model with probabilistic networks as ensemble members. The
            model has to be built but not compiled.
        :param optimizer: The optimizer wrapper with necessary specifications for compiling and
            training the model. Defaults to :class:`~trieste.models.optimizer.KerasOptimizer` with
            :class:`~tf.optimizers.Adam` optimizer, negative log likelihood loss and a dictionary
            of default arguments for Keras `fit` method: 1000 epochs, batch size 16, early stopping
            callback with patience of 50, and verbose 0.
            See https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
            arguments.
        :param bootstrap: Sample with replacement data for training each network in the ensemble.
            By default set to `False`.
        :raise ValueError: If ``model`` is not an instance of
            :class:`~trieste.models.keras.KerasEnsemble` or ensemble has less than two base
            learners (networks).
        """
        if model.ensemble_size < 2:
            raise ValueError(f"Ensemble size must be greater than 1 but got {model.ensemble_size}.")

        super().__init__(optimizer)

        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": 1000,
                "batch_size": 16,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=50, restore_best_weights=True
                    ),
                ],
            }

        
        if self.optimizer.loss is None:
            self.optimizer.loss = negative_log_likelihood

        model.model.compile(
            self.optimizer.optimizer,
            loss=[self.optimizer.loss] * model.ensemble_size,
            metrics=[self.optimizer.metrics] * model.ensemble_size,
        )

        self._model = model
        self._bootstrap = bootstrap

    def __repr__(self) -> str:
        """"""
        return f"DeepEnsemble({self.model!r}, {self.optimizer!r}, {self._bootstrap!r})"

    @property
    def model(self) -> tf.keras.Model:
        """ " Returns compiled Keras ensemble model."""
        return self._model.model

    @property
    def ensemble_size(self) -> int:
        """
        Returns the size of the ensemble, that is, the number of base learners or individual neural
        network models in the ensemble.
        """
        return self._model.ensemble_size

    def sample_index(self, size: int = 1) -> TensorType:
        """
        Returns a network index sampled randomly with replacement.
        """
        network_index = tf.random.uniform(
            shape=(tf.cast(size, tf.int32),), maxval=self.ensemble_size, dtype=tf.int32
        )
        return network_index

    def prepare_dataset(
        self, dataset: Dataset
    ) -> tuple[Dict[str, TensorType], Dict[str, TensorType]]:
        """
        Transform ``dataset`` into inputs and outputs with correct names that can be used for
        training the :class:`KerasEnsemble` model.

        If ``bootstrap`` argument in the :class:`~trieste.models.keras.DeepEnsemble` is set to
        `True`, data will be additionally sampled with replacement, independently for
        each network in the ensemble.

        :param dataset: A dataset with ``query_points`` and ``observations`` tensors.
        :return: A dictionary with input data and a dictionary with output data.
        """
        inputs = {}
        outputs = {}
        for index in range(self.ensemble_size):
            if self._bootstrap:
                resampled_data = sample_with_replacement(dataset)
            else:
                resampled_data = dataset
            input_name = self.model.input_names[index]
            output_name = self.model.output_names[index]
            inputs[input_name], outputs[output_name] = resampled_data.astuple()

        return inputs, outputs

    def prepare_query_points(self, query_points: TensorType) -> Dict[str, TensorType]:
        """
        Transform ``query_points`` into inputs with correct names that can be used for
        predicting with the model.

        :param query_points: A tensor with ``query_points``.
        :return: A dictionary with query_points prepared for predictions.
        """
        inputs = {}
        for index in range(self.ensemble_size):
            inputs[self.model.input_names[index]] = query_points

        return inputs

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance at ``query_points`` for the whole ensemble.

        Following <cite data-cite="lakshminarayanan2017simple"/> we treat the ensemble as a
        uniformly-weighted Gaussian mixture model and combine the predictions as

        .. math:: p(y|\mathbf{x}) = M^{-1} \Sum_{m=1}^M \mathcal{N}
            (\mu_{\theta_m}(\mathbf{x}),\,\sigma_{\theta_m}^{2}(\mathbf{x}))

        We further approximate the ensemble prediction as a Gaussian whose mean and variance
        are respectively the mean and variance of the mixture, given by

        .. math:: \mu_{*}(\mathbf{x}) = M^{-1} \Sum_{m=1}^M \mu_{\theta_m}(\mathbf{x})

        .. math:: \sigma^2_{*}(\mathbf{x}) = M^{-1} \Sum_{m=1}^M (\sigma_{\theta_m}^{2}(\mathbf{x})
            + \mu^2_{\theta_m}(\mathbf{x})) - \mu^2_{*}(\mathbf{x})

        This method assumes that the final layer in each member of the ensemble is
        probabilistic, an instance of :class:`¬tfp.distributions.Distribution`. In particular, given
        the nature of the approximations stated above the final layer should be a Gaussian
        distribution with `mean` and `variance` methods.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        query_points_transformed = self.prepare_query_points(query_points)

        ensemble_distributions = self.model(query_points_transformed)
        predicted_means = tf.math.reduce_mean(
            [dist.mean() for dist in ensemble_distributions], axis=0
        )
        predicted_vars = (
            tf.math.reduce_mean(
                [dist.variance() + dist.mean() ** 2 for dist in ensemble_distributions], axis=0
            )
            - predicted_means ** 2
        )

        return predicted_means, predicted_vars

    def predict_ensemble(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Returns mean and variance at ``query_points`` for each member of the ensemble. First tensor
        is the mean and second is the variance, where each has shape [..., M, N, 1], where M is
        the ``ensemble_size``.

        This method assumes that the final layer in each member of the ensemble is
        probabilistic, an instance of :class:`¬tfp.distributions.Distribution`, in particular
        `mean` and `variance` methods should be available.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points`` for each member of the ensemble.
        """
        query_points_transformed = self.prepare_query_points(query_points)

        ensemble_distributions = self.model(query_points_transformed)
        predicted_means = tf.convert_to_tensor([dist.mean() for dist in ensemble_distributions])
        predicted_vars = tf.convert_to_tensor([dist.variance() for dist in ensemble_distributions])

        return predicted_means, predicted_vars

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. We use the mixture approximation in
        :meth:`predict` for ``query_points`` and sample ``num_samples`` times from a Gaussian
        distribution given by the predicted mean and variance.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """

        predicted_means, predicted_vars = self.predict(query_points)
        normal = tfp.distributions.Normal(predicted_means, tf.sqrt(predicted_vars))
        samples = normal.sample(num_samples)

        return samples  # [num_samples, len(query_points), 1]

    def sample_ensemble(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. Each sample is taken from a Gaussian
        distribution given by the predicted mean and variance of a randomly chosen network in the
        ensemble. This avoids using the Gaussian mixture approximation and samples directly from
        individual Gaussian distributions given by each network in the ensemble.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """
        predicted_means, predicted_vars = self.predict_ensemble(query_points)

        stacked_samples = []
        for _ in range(num_samples):
            network_index = self.sample_index(1)[0]
            normal = tfp.distributions.Normal(
                predicted_means[network_index], tf.sqrt(predicted_vars[network_index])
            )
            samples = normal.sample()
            stacked_samples.append(samples)

        samples = tf.stack(stacked_samples, axis=0)
        return samples  # [num_samples, len(query_points), 1]

    def trajectory_sampler(self) -> TrajectorySampler[DeepEnsemble]:
        """
        Return a trajectory sampler. For :class:`DeepEnsemble`, we use an ensemble
        sampler that randomly picks a network from the ensemble and uses its predicted means
        for generating a trajectory.

        :return: The trajectory sampler.
        """
        return EnsembleTrajectorySampler(self)

    def update(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to update data.
        `TrainableProbabilisticModel` interface, however, requires an update method, so
        here we simply pass the execution.
        """
        pass

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the underlying Keras ensemble model with the specified ``dataset``.

        Optimization is performed by using the Keras `fit` method, rather than applying the
        optimizer and using the batches supplied with the optimizer wrapper. User can pass
        arguments to the `fit` method through ``minimize_args`` argument in the optimizer wrapper.
        These default to using 100 epochs, batch size 100, and verbose 0. See
        https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
        arguments.

        Note that optimization does not return the result, instead optimization results are
        stored in a history attribute of the model object.

        :param dataset: The data with which to optimize the model.
        """

        x, y = self.prepare_dataset(dataset)
        self.model.fit(x=x, y=y, **self.optimizer.fit_args)


class DeepEvidentialRegression(
    KerasPredictor, EvidentialPriorModel, TrainableProbabilisticModel
):
    """
    A :class:`~trieste.model.TrainableProbabilisticModel` wrapper for a deep evidential model 
    built using Keras.

    Deep evidential regression is a deterministic deep neural network that seeks to learn the 
    parameters to the posterior higher order deep evidential distributions and has good 
    quantifications of uncertainty at fast speeds in practice (<cite data-cite="amini2020evidential"/>). 
    Furthermore, the deep evidential model can easily separate between aleatoric and epistemic uncertainty.  
    The model consists of a simple fully connected feed forward network whose final output layer is 
    configured to output the necessary evidential parameters. The model trains using a combination 
    of the negative log-likelihood of the Normal Inverse Gamma distribution and a custom regularizer 
    (<cite data-cite="amini2020evidential"/>) that makes the problem well defined. 

    The dual loss functions are controlled by a single weight coefficient, ``reg_weight`` and although
    the original paper does not explicitly note the use of an iterative search procedure to optimize
    this parameter, the author's original code does, and we include its use here. In practice, it 
    improves performance of the model and makes it less sensitive to the hyperparameter choice. 

    We provide classes for constructing the base network using Keras
    (:class:`~trieste.models.keras.DeepEvidentialNetwork`) in the `architectures` package that should 
    be used with the :class:`~trieste.models.keras.DeepEvidentialRgression` wrapper. We also provide 
    the necessary loss functions and a custom callback to implement the iterative procedures for 
    computing the loss in the `utils` package. These methods are implented by default in the model wrapper. 

    Note that currently we do not support setting up the model with dictionary configs and saving
    the model during Bayesian optimization loop (``track_state`` argument in
    :meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method should be set to `False`).
    """
    def __init__(
        self,
        model: DeepEvidentialNetwork,
        optimizer: Optional[KerasOptimizer] = None,
        reg_weight: float = 0.,
        maxi_rate: float = 1e-4,
        epsilon: float = 1e-2,
        verbose: int = 0 #Temporary parameter to be used with Callback for diagnosing in development.
    ) -> None:
        
        """
        :param model: A Keras model built to output evidential parameters: an instance of 
            :class:`trieste.models.keras.DeepEvidentialNetwork`. Themodel has to be built but 
            not compiled.
        :param optimizer: The optimizer wrapper with necessary specifications for compiling and
            training the model. Loss function passed with this optimizer will be ignored and will
            instead use a weighted combination, controlled by ``reg_weight``, of 
            :function:`~trieste.models.keras.utils.normal_inverse_gamma_log_likelihood` and
            :function:`~trieste.models.keras.utils.normal_inverse_gamma_regularizer`. The constructor
            will also add :class:`~trieste.models.keras.utils.DeepEvidentialCallback` to the optimizer, 
            which is required to run this model. This callback is not meant to be instantiated outside 
            of this model's constructor. Otherwise the optimizer defaults to 
            :class:`~trieste.models.optimizer.KerasOptimizer` with :class:`~tf.optimizers.Adam` 
            optimizer, and a dictionary of default arguments for Keras `fit` method: 1000 epochs, 
            batch size 16, early stopping callback with patience of 100, resotre_best_weights True, 
            and verbose 0. See https://keras.io/api/models/model_training_apis/#fit-method for a list 
            of possible arguments.
        :param reg_weight: The weight attributed to the regularization loss that trades off between 
            uncertainty inflation and model fit. Smaller values lead to higher degrees of confidence, 
            whereas larger values lead to inflation of uncertainty. A fixed value of around 0.01 
            seems to work well for small datasets, but ``reg_weight`` defualts to 0 to allow for 
            an automatic incremental search for the best value using ``maxi_rate``.
        :param maxi_rate: Throughout training, the ``reg_weight`` is automatically adjusted based on 
            previous outputs of the regularization loss. This update is applied at the end of every 
            batch by: ``reg_weight`` += ``maxi_rate`` * (regularization loss - ``epsilon``). A default
            of 1e-4 in conjunction with a ``reg_weight`` of 0 and ``epsilon`` of 0.01 seems to work well. 
        :param epsilon: A parameter used in updating ``reg_weight`` throughout training as described above.
        """

        super().__init__(optimizer)

        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": 1000,
                "batch_size": 16,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=100, restore_best_weights=True
                    )
                ],
            }

        self.reg_weight = tf.Variable(reg_weight, dtype=model.layers[-1].dtype)
        self.epsilon = epsilon
        self.maxi_rate = maxi_rate
        self.verbose = verbose
        
        try:
            if not isinstance(self.optimizer.fit_args["callbacks"], list):
                if isinstance(self.optimizer.fit_args["callbacks"], Sequence):
                    self.optimizer.fit_args["callbacks"] = list(self.optimizer.fit_args["callbacks"])
                else:
                    self.optimizer.fit_args["callbacks"] = [self.optimizer.fit_args["callbacks"]]
            self.optimizer.fit_args["callbacks"].append(
                DeepEvidentialCallback(
                    self.reg_weight, self.maxi_rate, self.epsilon, self.verbose
                )
            )
        except KeyError:
            self.optimizer.fit_args["callbacks"] = [
                    DeepEvidentialCallback(
                        self.reg_weight, self.maxi_rate, self.epsilon, self.verbose
                    )
                ]

        model.compile(
            self.optimizer.optimizer,
            loss=[
                normal_inverse_gamma_negative_log_likelihood, 
                normal_inverse_gamma_regularizer
            ],
            loss_weights = [1., self.reg_weight],
            metrics=[self.optimizer.metrics],
        )

        self._model = model
        self._learning_rate = self.optimizer.optimizer.learning_rate.numpy()
    
    @property
    def model(self) -> tf.keras.Model:
        """ Returns compiled Keras ensemble model."""
        return self._model

    def __repr__(self) -> str:
        return f"DeepEvidentialRegression({self.model!r}, {self.optimizer!r})"


    def sample_normal_parameters(
        self, 
        gamma: TensorType,
        v: TensorType,
        alpha: TensorType,
        beta: TensorType, 
        num_samples:int
    ) -> tuple[TensorType, TensorType]:
        """
        Returns a tensor of means and a tensor of variances that parametrized the
        posterior Gaussian distribution of our outputs. We use the evidential parameters
        gamma, v, alpha, beta to sample from a Gaussian distribution to sample our means
        and an Inverse Gamma distribution sample our variances.

        :param gamma: the mean of the evidential Gaussian distribution.
        :param v: sigma/v parameterizes the variance of the evidential Gaussian distribution.
        :param alpha: the concentration (shape) of the evidential Inverse Gamma distribution.
        :param beta: the scale of the evidential Inverse Gamma distribution.
        :num_samples: number of samples: S. 

        :return: mean and variance tensors with shape [S, N, 1] each. 
        """

        sigma_dist = tfp.distributions.InverseGamma(alpha, beta)
        sigma_samples = sigma_dist.sample(num_samples)
        
        mu_dist = tfp.distributions.Normal(gamma, (sigma_samples/v)**0.5)

        mu_samples = mu_dist.sample(1)
        mu_samples = tf.reshape(mu_samples, sigma_samples.shape)
        
        return mu_samples, sigma_samples # [num_samples, len(query_points), 1] x2

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. We use :meth:`predict` for 
        ``query_points`` to get the evidential parameters. We use :meth:`sample_normal_parameters`
        to sample mu and sigma tensors from our evidential distributions to parametrize
        our posterior Gaussian distribution. We draw our samples from this Gaussian distribution.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point: S. 
        :return: The samples. This has shape [S, N, 1]
        """
        evidential_output = self.model(query_points)[0]
        gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)

        mu, sigma = self.sample_normal_parameters(gamma, v, alpha, beta, num_samples)
       
        observation_dist = tfp.distributions.Normal(mu, sigma**0.5)
        samples = observation_dist.sample(1)
        
        samples = tf.reshape(tf.squeeze(samples), (num_samples, len(query_points), 1))
        
        return samples # [num_samples, len(query_points), 1]


    def update(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to update data.
        `TrainableProbabilisticModel` interface, however, requires an update method, so
        here we simply pass the execution.
        """
        pass
    
    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the underlying Keras ensemble model with the specified ``dataset``.

        Optimization is performed by using the Keras `fit` method, rather than applying the
        optimizer and using the batches supplied with the optimizer wrapper. User can pass
        arguments to the `fit` method through ``fit_args`` argument in the optimizer wrapper.
        These default to using 1000 epochs, batch size 16, and verbose 0 as well as a custom
        loss function and callback necessary for this model described in the constructor. See
        https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
        arguments.

        Note that optimization does not return the result, instead optimization results are
        stored in a history attribute of the model object. Optimization fits to two copies of
        the dataset's observations to leverage a dynamic weighting of the loss function and its
        regularizer. The history attribute contains three losses where ``loss`` referes to the 
        combined weighted loss, ``output_1_loss`` refers to the loss computed by 
        :function:`~trieste.keras.models.utils.normal_inverse_gamma_negative_log_likelihood` 
        and ``output_2_loss`` refers to the loss computed using 
        :function:`~trieste.keras.models.utils.normal_inverse_gamma_regularizer`.

        :param dataset: The data with which to optimize the model.
        """

        x, y = dataset.astuple()
        self.model.fit(x, [y, y], **self.optimizer.fit_args)
        self.optimizer.optimizer.learning_rate.assign(self._learning_rate)
    

    def predict(self, query_points: TensorType, aleatoric: bool = False) -> tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance at ``query_points`` for the model.

        Following <cite data-cite="amini2020evidential"/> we use the evidential parameters outputted
        to create our evidential distributions:

        .. math:: Y ~ \mathcal{N}(\mu, \sigma^{2})
        .. math:: \mu ~ \mathcal{N}(\gamma, \sigma^{2} \nu^{-1})
        .. math:: \sigma^{2} ~ \GAMMA^{-1} (\alpha, \beta)

        The `mean` of the distribution is simply mu whose expectation will be gamma. Deep evidential
        regression is able to distinguish between epistemic and aleatoric uncertainty. The aleatoric
        uncertainty is given by:

        .. math:: \mathbf{E}[\sigma^{2}] = \frac{\beta}{\alpha - 1}

        and the epistemic uncertainty is given by:

        ..math:: \mathbf{Var}[\mu] = \frac{\beat}{\nu(\alpha - 1)}

        By default the predict method outputs the epistemic uncertainty only. The aleatoric uncertainty 
        can be added using the ``aleatoric`` argument.

        :param query_points: The points at which to make predictions.
        :param aleatoric: If false outputs only the epistemic uncertainty. If true the aleatoric 
            uncertainty is added. 
        :return: The predicted mean and uncertainty of the observations at the specified
            ``query_points``.
        """

        evidential_output = self.model(query_points)[0]
        gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
        
        epistemic = beta / ((alpha - 1) * v)
        uncertainty = epistemic + beta/(alpha-1) if aleatoric else epistemic
        
        return gamma, uncertainty