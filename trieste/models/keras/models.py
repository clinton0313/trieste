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

from typing import Any, Dict, Optional, Sequence
from copy import deepcopy

import dill
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.callbacks import Callback

from ...data import Dataset
from ...types import TensorType
from ..interfaces import (
    EnsembleModel,
    HasTrajectorySampler,
    TrainableProbabilisticModel,
    TrajectorySampler,
)
from ..optimizer import KerasOptimizer
from .architectures import KerasEnsemble, MultivariateNormalTriL
from .interface import KerasPredictor
from .sampler import EnsembleTrajectorySampler
from .utils import negative_log_likelihood, sample_with_replacement, KernelDensityEstimator

import datetime

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
                    )
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

    def __getstate__(self) -> dict[str, Any]:
        # When pickling use to_json to save any optimizer fit_arg callback models
        state = self.__dict__.copy()
        if self._optimizer:
            # jsonify all the callback models, pickle the optimizer(!), and revert (ugh!)
            callback: Callback
            saved_models: list[KerasOptimizer] = []
            for callback in self._optimizer.fit_args["callbacks"]:
                saved_models.append(callback.model)
                callback.model = callback.model and callback.model.to_json()
            state["_optimizer"] = dill.dumps(state["_optimizer"])
            for callback, model in zip(self._optimizer.fit_args["callbacks"], saved_models):
                callback.model = model

        return state


    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore optimizer and callback models after depickling, and recompile.
        self.__dict__.update(state)
        if self._optimizer:
            # unpickle the optimizer, and restore all the callback models
            self._optimizer = dill.loads(self._optimizer)
            for callback in self._optimizer.fit_args.get("callbacks", []):
                if callback.model:
                    callback.model = tf.keras.models.model_from_json(
                        callback.model,
                        custom_objects={"MultivariateNormalTriL": MultivariateNormalTriL},
                    )
        # Recompile the model
        self._model.model.compile(
            self.optimizer.optimizer,
            loss=[self.optimizer.loss] * self._model.ensemble_size,
            metrics=[self.optimizer.metrics] * self._model.ensemble_size,
            )

class DirectEpistemicUncertaintyPredictor(
    KerasPredictor, TrainableProbabilisticModel
):
    """
    A :class:`~trieste.model.TrainableProbabilisticModel` wrapper for a direct epistemic uncertainty
    prediction (DEUP) model built using Keras.

    The method employs an auxiliary deep neural network to predict the uncertainty that stems
    from the generalization error of the main predictor, which in principle is reducible with more
    data and higher effective capacity. Unlike other available Keras models, which solely exploit 
    model variance to approximate the total uncertainty, DEUP accounts for the bias induced in
    training neural networks with limited data, which can induce a preference on the functions it
    learns away from the Bayes-optimal predictor (<cite data-cite="jain2022"/>). The main model,
    `f_model`, is trained to predict outcomes of the function of interest and is built as an
    ensemble of deep neural networks using the :class:`~trieste.models.keras.DeepEvidentialRgression` wrapper.
    each a fully connected multilayer probabilistic network. The auxiliary model, `e_model`, is trained.
    to predict the squared loss of the main predictor, which in the regression setup can be shown to 
    approximate the total uncertainty stemming both from the model variance and the potential misspecification 
    caused by, for example, early stopping.

    A particular advantage of training the auxiliary model is that the error predictor can be explicitly
    trained to account for examples that may come from a distribution different from the distribution of
    most of the training examples. These non-stationary settings, likely in the context of Bayesian 
    optimization, make it challenging to train the error predictor, as the measured error around a parameter
    combination will differ before and after the incorporation of the queried set of arguments to the training set.
    We account for this by using additional features as input to the error predictor, namely the log-density
    of the function parameters and the model variance estimates. The former is computed using kernel density
    estimation and assuming a Gaussian kernel, while the latter is computed from the variance estimates of the 
    main deep ensemble model.  

    In practice, we provide a warm start to the error predictor by creating multiple versions of the main
    predictor trained on different subsets of the training data. This approach, inspired by standard cross
    validation, builds a larger set of targets for the error predictor and avoids discarding valuable observations
    in the early training of the error predictor. The warm start is enabled by default, and can be disabled by
    setting the ``init_buffer`` argument to False.

    Note that currently we do not support setting up the model with dictionary configs and saving
    the model during Bayesian optimization loop (``track_state`` argument in
    :meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method should be set to `False`).
    """
    def __init__(
        self,
        model: Sequence[dict[str, Any]],
        optimizer: Optional[KerasOptimizer] = None,
        init_buffer: bool = False
    ) -> None:

        """
        :param model: A dictionary with two models: the main predictor and the auxiliary error model.
            The main Keras model should be a compiled model of class :class:`trieste.models.DeepEnsemble`
            or class :class:`trieste.models.keras.MonteCarloDropout`. The auxiliary model is an instance of
            :class:`trieste.models.keras.EpistemicUncertaintyPredictor`. The model has to be built but not
            compiled.
        :param optimizer: The optimizer wrapper with necessary specifications for compiling and
            training the model. Defaults to :class:`~trieste.models.optimizer.KerasOptimizer` with
            :class:`~tf.optimizers.Adam` optimizer, mean squared error loss and a dictionary
            of default arguments for Keras `fit` method: 1000 epochs, batch size 16, early stopping
            callback with patience of 50, and verbose 0. 
            See https://keras.io/api/models/model_training_apis/#fit-method for a list 
            of possible arguments.
        :param init_buffer: A boolean that enables the pre-training of the error predictor by training several
            main models using cross-validated slices of the dataset and fitting the resulting models on
            the entire dataset. This constructs a dataset of [N*K, D] observations nad squared losses, which
            are used to train the auxiliary predictor.
        """

        super().__init__(optimizer)

        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": 1000,
                "batch_size": 32,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=100, restore_best_weights=True
                    )
                ],
            }

        self.optimizer.loss = "mse"

        self._learning_rate = self.optimizer.optimizer.learning_rate.numpy()
        
        model["e_model"].compile(
            self.optimizer.optimizer,
            loss=[self.optimizer.loss],
            metrics=[self.optimizer.metrics],
        )

        self._model = model
        self._init_buffer = init_buffer
        self._data_u = None     # [DAV] uncertainty dataset
        self._prior_size = None # [DAV] track new observations

    def __repr__(self) -> str:
        """"""
        return f"DirectEpistemicUncertaintyPredictor({self.model!r}, {self.optimizer!r})"

    @property
    def model(self) -> tuple[DeepEnsemble, tf.keras.Model]:
        """
        Returns two compiled models: A Keras ensemble model and an epistemic uncertainty predictor.
        """
        assert issubclass(type(self._model["f_model"]), TrainableProbabilisticModel), "[DAV]"
        assert issubclass(type(self._model["e_model"]), tf.keras.Model), "[DAV]"
        return self._model["f_model"], self._model["e_model"]

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

    def update(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to update data.
        `TrainableProbabilisticModel` interface, however, requires an update method, so
        here we simply pass the execution.
        """
        pass

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the underlying Direct Epistemic Uncertainty Prediction model with the 
        specified ``dataset``.

        Optimization is performed by using the Keras `fit` method, rather than applying the
        optimizer and using the batches supplied with the optimizer wrapper. User can pass
        arguments to the `fit` method through ``minimize_args`` argument in the optimizer wrapper.
        These default to using 100 epochs, batch size 32, and verbose 0. See
        https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
        arguments.

        Note that optimization does not return the result, instead optimization results are
        stored in a history attribute of the model object. The algorithm iterates
        over two copies of the dataset's observations to account for the
        stationarizing features and the two targets: the target outcome for the main
        predictor and the squared loss for the auxiliary predictor. The procedure follows
        the main proposed algorithm in <cite data-cite="jain2022"/>.

        :param dataset: The data with which to optimize the model.
        """     
        # optional init buffer
        if self._data_u is None: 
            self._prior_size = dataset.query_points.shape[0]
            self.density_estimator = KernelDensityEstimator(kernel="gaussian")
            if self._init_buffer:
                print("Access uncertainty buffer", datetime.datetime.now())
                self._data_u = self.uncertainty_buffer(dataset=dataset, iterations=1)
            else:
                self._data_u = Dataset(
                    tf.zeros([0, dataset.query_points.shape[-1] + 2], dtype=tf.float64), 
                    tf.zeros([0, 1], dtype=tf.float64)
                )
        
        print("optim loop", datetime.datetime.now(), self._prior_size)

        x, y = dataset.astuple()

        # post-oracle, pre-refit append
        self._data_u = self.data_u_appender(x, y)

        # post-oracle, post-refit
        self.model[0].optimize(dataset)
        self.density_estimator.fit(dataset.query_points)
        self._data_u = self.data_u_appender(x, y)

        xu, yu = self._data_u.astuple()

        # pre-new-oracle, fit u (it is empty if not init_buffering at the moment)
        try:
            self.model[1].fit(xu, yu, **self.optimizer.fit_args)
        except:
            pass
        self.optimizer.optimizer.learning_rate.assign(self._learning_rate)

        # increase "seen" observations (only after first set of candidates)
        self._prior_size = dataset.query_points.shape[0]

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance at ``query_points`` for the Direct Epistemic
        Uncertainty Prediction model.

        Following <cite data-cite="jain2022"/>, we consider the case for a model with
        a Gaussian ground truth, i.e.
                .. math:: p(y \mid x)=\mathcal{N}\left(y ; f^{*}(x), \sigma^{2}(x)\right).

        In this scenario, it can be shown that the epistemic uncertainty can be estimated
        as the squared difference between main-model predictions and the Bayes optimal
        prediction at a given point, that is:
                .. math:: \mathcal{E}(\hat{f}, x)=\left(\hat{f}(x)-f^{*}(x)\right)^{2}

        As the Bayes optimal is unattainable, we instead approximate the epistemic uncertainty
        by computing the total uncertainty of the model, which is captured by the expected
        squared loss of the main-model predictions. This can be deconstructed as follows
                .. math:: \mathcal{U}(\hat{f}, x)=E_{P(. \mid x)}\left[(\hat{f}(x)-y)^{2}\right]=\left(\hat{f}(x)-f^{*}(x)\right)^{2}+\sigma^{2}(x)
        
        The predict method returns the point estimates of the main predictor, and uses point estimates
        of the squared loss to return predictions for model uncertainty. Note that the dataset used to
        train the models in a BO setting grows over time, which renders estimates of uncertainty
        as iteration dependent. Following the original paper, we account for this evolving uncertainty 
        space by considering stationarizing variables, namely the predicted variance from the ensemble 
        model and the Gaussian density of our queried points. These are used to enhance the dataset of
        features the auxiliary model predicts on, so that a D-dimensional queried point will be passed
        as a [D+2,1] observation to the error predictor.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        if not tf.is_tensor(query_points):
            query_points = tf.convert_to_tensor(query_points)
        if query_points.shape.rank == 1:
            query_points = tf.expand_dims(query_points, axis=-1)

        f_pred, f_var = self.model[0].predict(query_points)
        density_scores = self.density_estimator.score_samples(query_points)
        data_u = tf.concat((query_points, f_var, density_scores), axis=1)
        e_pred = self.model[1](data_u)
        return f_pred, e_pred

    def __copy__(self):
        r"""
        Creates a copy of the model used in the initial buffering of the data for the 
        uncertainty predictor.
        """
        return DirectEpistemicUncertaintyPredictor(model=self._model)

    def data_u_appender(self, query_points, observations):
        """
        Enhances the new queried observations with estimates of their Gaussian density
        and main model variance, and appends them as new observations to optimize the
        auxiliary model.
        """
        new_points = query_points[self._prior_size:,:]
        new_observations = observations[self._prior_size:,:]

        # stationarizing feature: variance
        f_pred, f_var = self.model[0].predict(new_points)
        
        # stationarizing feature: density
        density_scores = self.density_estimator.score_samples(new_points)

        new_data_u = Dataset(
            tf.concat((new_points, f_var, density_scores), axis=1),   # [N, D+2]
            tf.pow(tf.subtract(f_pred, new_observations), 2)
        )
        return Dataset(
            (self._data_u + new_data_u).query_points,
            (self._data_u + new_data_u).observations,            

        )

    def uncertainty_buffer(self, dataset: Dataset, iterations: int) -> Dataset:
        """
        Builds an initial dataset of observations to kickstart the error predictor. The 
        scheme employs cross-validation to train multiple main models using partial datasets
        of the initial data points, and predicts both in sample and out of sample outcomes.
        The squared loss of these predictions are used to construct the target dataset for
        the error predictor.
        """
        points, targets = [], []

        data = tf.concat((dataset.query_points, dataset.observations), axis=1)
        for _ in tf.range(iterations):
            for set in tf.random.shuffle(tf.split(data, 2, axis=0)):
                data_ = Dataset(set[:,:-1], tf.expand_dims(set[:,-1], axis=1))

                # stationarizing feature: variance
                f_ = self.__copy__().model[0]
                f_.optimize(data_)
                f_pred, f_var = f_.predict(dataset.query_points)

                # stationarizing feature: density
                self.density_estimator.fit(dataset.query_points)
                density_scores = self.density_estimator.score_samples(dataset.query_points)

                targets.append(tf.pow(tf.subtract(f_pred, dataset.observations), 2))
                points.append(tf.concat((dataset.query_points, f_var, density_scores), axis=1)) # [DAV] manually add f_var here, need to fix
        
        points, targets = tf.concat(points, axis=0), tf.concat(targets, axis=0)
        return Dataset(points, targets)