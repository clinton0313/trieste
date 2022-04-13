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

from ...types import TensorType
from ..interfaces import (
    EnsembleModel,
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


class DeepEvidentialTrajectorySampler(TrajectorySampler[TrainableProbabilisticModel]):
    """
    This class builds functions that approximate a trajectory by taking a draw from the posterior
    distribution.
    """

    def __init__(self, model: TrainableProbabilisticModel):
        """
        :param model: The Deep Evidential model to sample from.
        """
        if not isinstance(model, TrainableProbabilisticModel):
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
            from the model, taking an input of shape `[N, 1, D]` and returning shape `[N, 1]`.
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

    def __init__(self, model: TrainableProbabilisticModel):
        """
        :param model: The model of the objective function.
        """
        self._model = model
        self._initialized = tf.Variable(False, trainable=False)

        self._batch_size = tf.Variable(
            0, dtype=tf.int32, trainable=False
        )

        self._seeds = tf.Variable(
            tf.ones([0,0], dtype=tf.int32), shape=[None, None], trainable=False
        )
            
    def __call__(self, x: TensorType) -> TensorType:  # [N, B, d] -> [N, B]
        """
        Call trajectory function. It uses `tf.random` seeds to fix the matrix of dropout
        inputs or weights in the kernel, and performs a single forward pass to return the
        equivalent to a posterior draw.
        """
        if not self._initialized:  # work out desired batch size from input
            self._batch_size.assign(tf.shape(x)[-2])  # B
            self.resample() # sample B seeds to fix
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

        predictions = []
        batch_index = tf.range(0, self._batch_size, 1)
        _ = self._model.sample(x[:1,0,:], num_samples=1) # [DAV] Somehow first seed doesn't propagate unless I've done a dummy pred before.
        for b, seed in zip(batch_index, tf.unstack(self._seeds)):
            tf.random.set_seed(seed) # [DAV] A possible local seed? One can pass operational seeds to both types of dropouts, but it doesn't seem to fix the prediction
            predictions.append(self._model.sample(x[:,b,:], num_samples=1)[0])

        return tf.transpose(tf.squeeze(predictions, axis=-1), perm=[1,0])

    def resample(self) -> None:
        """
        Efficiently resample in-place without retracing.
        """
        self._seeds.assign(tf.random.uniform(shape=(self._batch_size, 1), minval=1, maxval=999999999, dtype=tf.int32))