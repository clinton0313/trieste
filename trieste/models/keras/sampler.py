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

"""
This module is the home of the sampling functionality required by some
of the Trieste's Keras model wrappers.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from ...types import TensorType
from ..interfaces import (
    EnsembleModel,
    EvidentialPriorModel,
    TrainableProbabilisticModel,
    TrajectoryFunction,
    TrajectoryFunctionClass,
    TrajectorySampler,
)


class EnsembleTrajectorySampler(TrajectorySampler[EnsembleModel]):
    """
    This class builds functions that approximate a trajectory by randomly choosing a network from
    the ensemble and using its predicted means as a trajectory.
    """

    def __init__(self, model: EnsembleModel):
        """
        :param model: The ensemble model to sample from.
        """
        if not isinstance(model, EnsembleModel):
            raise NotImplementedError(
                f"EnsembleTrajectorySampler only works with EnsembleModel models, that support "
                f"ensemble_size, sample_index, predict_ensemble and sample_ensemble methods; "
                f"received {model.__repr__()}"
            )

        super().__init__(model)

        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._model!r}"

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by randomly choosing a network from
        the ensemble and using its predicted means as a trajectory.

        :return: A trajectory function representing an approximate trajectory
            from the model, taking an input of shape `[N, 1, D]` and returning shape `[N, 1]`.
        """
        return ensemble_trajectory(self._model)

    def resample_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently resample a :const:`TrajectoryFunction` in-place to avoid function retracing
        with every new sample.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        tf.debugging.Assert(isinstance(trajectory, ensemble_trajectory), [])
        trajectory.resample()  # type: ignore
        return trajectory


class ensemble_trajectory(TrajectoryFunctionClass):
    """
    Generate an approximate function draw (trajectory) by randomly choosing a network from
    the ensemble and using its predicted means as a trajectory.
    """

    def __init__(self, model: EnsembleModel):
        """
        :param model: The model of the objective function.
        """
        self._model = model
        self._network_index = tf.Variable(self._model.sample_index(1)[0])

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:  # [N, 1, d] -> [N, 1]
        """Call trajectory function."""
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This trajectory only supports batch sizes of one.",
        )
        x = tf.squeeze(x, -2)  # [N, D]
        return self._model.predict_ensemble(x)[0][self._network_index]

    def resample(self) -> None:
        """
        Efficiently resample in-place without retracing.
        """
        self._network_index.assign(self._model.sample_index(1)[0])


class DeepEvidentialTrajectorySampler(TrajectorySampler[EvidentialPriorModel]):
    """
    This class builds functions that approximate a trajectory by taking a draw from the posterior
    distribution for a :class:~trieste.models.keras.models.DeepEvidentialRegression model. 
    """

    def __init__(self, model: EvidentialPriorModel):
        """
        :param model: The Deep Evidential model to sample from.
        """
        if not isinstance(model, EvidentialPriorModel):
            raise NotImplementedError(
                f"DeepEvidentialTrajectorySampler only works with Deep Evidential Regression Model; "
                f"received {model.__repr__()}"
            )

        super().__init__(model)

        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._model!r}"

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by using the predicted means
    of the stochastic forward passes as a trajectory.

        :return: A trajectory function representing an approximate trajectory
            from the model, taking an input of shape `[N, B, D]` and returning shape `[N, B]`. 
            N is the number of samples in each batch and B is the batch size. 
        """

        return deep_evidential_trajectory(self._model)

    def resample_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently resample a :const:`TrajectoryFunction` in-place to avoid function retracing
        with every new sample.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        tf.debugging.Assert(isinstance(trajectory, deep_evidential_trajectory))
        trajectory.resample()
        return trajectory

class deep_evidential_trajectory(TrajectoryFunctionClass):
    """
    Generate an approximate function draw (trajectory) by drawing a mean and variance
    vector from the posterior distributions.
    """

    def __init__(self, model: EvidentialPriorModel):
        """
        :param model: The model of the objective function.
        """
        self._model = model
        self._initialized = tf.Variable(False, trainable=False)

        self._batch_size = tf.Variable(
            0, dtype=tf.int32, trainable=False
        )
        self._evidential_parameters = None
        self._mu = tf.Variable(0, trainable=False)
        self._sigma = tf.Variable(0, trainable=False)
            
    def __call__(self, x: TensorType) -> TensorType:  # [N, B, d] -> [N, B]
        """
        Call trajectory function. Makes a draw from the posterior normal distribution
        of outputs given the predicted evidential parameters. 
        """
        if not self._initialized:  # work out desired batch size from input
            self._batch_size.assign(tf.shape(x)[-2])  # B
            self._evidential_parameters = self._model(x)
            self.resample() # Draws a mu and sigma vector
            self._initialized.assign(True)

        tf.debugging.assert_equal(
            tf.shape(x)[-2],
            self._batch_size.value(),
            message=f"""
            This trajectory only supports batch sizes of {self._batch_size}.
            If you wish to change the batch size you must get a new trajectory
            by calling the get_trajectory method of the trajectory sampler.
            """
        )

        dist = tfp.distributions.Normal(self._mu, self._sigma)
        predictions = dist.sample(1)

        return tf.reshape(predictions, (-1, self._batch_size))

    def resample(self) -> None:
        """
        Efficiently resample in-place without retracing. By redrawing mu and sigma vectors based off
        of already saved evidential parameters.
        """
        tf.debugging.Assert(
            self._evidential_parameters is not None,
            message="Trajectory sampler needs to have been called ot initialize before resampling."
        )
        gamma, lamb, alpha, beta = tf.split(self._evidential_parameters, 4, axis = -1)
        self._mu, self._sigma = self._model.sample_normal_parameters(gamma, lamb, alpha, beta, 1)