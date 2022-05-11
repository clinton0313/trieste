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

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


from ...data import Dataset
from ...types import TensorType


def get_tensor_spec_from_data(dataset: Dataset) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    r"""
    Extract tensor specifications for inputs and outputs of neural network models, based on the
    dataset. This utility faciliates constructing neural networks, providing the required
    dimensions for the input and the output of the network. For example

    >>> data = Dataset(
    ...     tf.constant([[0.1, 0.2], [0.3, 0.4]]),
    ...     tf.constant([[0.5], [0.7]])
    ... )
    >>> input_spec, output_spec = get_tensor_spec_from_data(data)
    >>> input_spec
    TensorSpec(shape=(2,), dtype=tf.float32, name='query_points')
    >>> output_spec
    TensorSpec(shape=(1,), dtype=tf.float32, name='observations')

    :param dataset: A dataset with ``query_points`` and ``observations`` tensors.
    :return: Tensor specification objects for the ``query_points`` and ``observations`` tensors.
    :raise ValueError: If the dataset is not an instance of :class:`~trieste.data.Dataset`.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"This function works only on trieste.data.Dataset objects, however got"
            f"{type(dataset)} which is incompatible."
        )
    input_tensor_spec = tf.TensorSpec(
        shape=(dataset.query_points.shape[1:]),
        dtype=dataset.query_points.dtype,
        name="query_points",
    )
    output_tensor_spec = tf.TensorSpec(
        shape=(dataset.observations.shape[1:]),
        dtype=dataset.observations.dtype,
        name="observations",
    )
    return input_tensor_spec, output_tensor_spec


def sample_with_replacement(dataset: Dataset) -> Dataset:
    """
    Create a new ``dataset`` with data sampled with replacement. This
    function is useful for creating bootstrap samples of data for training ensembles.

    :param dataset: The data that should be sampled.
    :return: A (new) ``dataset`` with sampled data.
    :raise ValueError (or InvalidArgumentError): If the dataset is not an instance of
        :class:`~trieste.data.Dataset` or it is empty.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"This function works only on trieste.data.Dataset objects, however got"
            f"{type(dataset)} which is incompatible."
        )
    tf.debugging.assert_positive(len(dataset), message="Dataset must not be empty.")

    n_rows = dataset.observations.shape[0]

    index_tensor = tf.random.uniform((n_rows,), maxval=n_rows, dtype=tf.dtypes.int32)

    observations = tf.gather(dataset.observations, index_tensor, axis=0)
    query_points = tf.gather(dataset.query_points, index_tensor, axis=0)

    return Dataset(query_points=query_points, observations=observations)


def negative_log_likelihood(
    y_true: TensorType, y_pred: tfp.distributions.Distribution
) -> TensorType:
    """
    Maximum likelihood objective function for training neural networks.

    :param y_true: The output variable values.
    :param y_pred: The output layer of the model. It has to be a probabilistic neural network
        with a distribution as a final layer.
    :return: Negative log likelihood values.
    """
    return -y_pred.log_prob(y_true)


class KernelDensityEstimator:
    """
    Kernel density estimator of the D-dimensional parameter space.
    """
    def __init__(self, kernel: str = 'gaussian'):
        self.kernel = kernel
        
    def fit(self, query_points: tf.Tensor, bandwidth: tf.Tensor = None):
        """
        Fits an optimal bandwidth given a number of query points. We use a grid search to find
        the optimal value in the bandwidth space, defined as the bandwidth that maximizes the
        in-sample posterior probability of observing the points. 

        Given a bandwidth, individual Gaussian distributions are constructed for each observation.

        :param query_points: The points to use while searching the optimal bandwidth.
        :param bandwidth: A fixed bandwidth can be instead provides, by default is None.
        """
        self._dtype = query_points.dtype
        if bandwidth is not None:
            self.bandwidth = bandwidth
        else:
            if query_points.shape[0] > 1:
                bandwidth_search_space = np.logspace(-3,1,50)
                grid = RandomizedSearchCV(
                    estimator=KernelDensity(kernel=self.kernel), 
                    param_distributions={"bandwidth": bandwidth_search_space}, 
                    n_iter=50,
                    cv=query_points.shape[0]
                )
                grid.fit(query_points.numpy())
                self.bandwidth = tf.constant([grid.best_estimator_.bandwidth], dtype=self._dtype)
            else:
                self.bandwidth = tf.constant([0.2], dtype=self._dtype)
        self.kernels = [
            tfp.distributions.MultivariateNormalDiag(loc=x, scale_identity_multiplier=self.bandwidth) 
            for x in query_points
        ]

    def score_samples(self, query_points):
        """
        Computes the kernel density estimation given the optimal bandwidth on data points and
        the distributions computed during fitting. The probability of a query point is aggregated
        across individual Gaussian distributions, which yields the density estimator.

        :param query_points: The [N, D] points to predict the density on.
        :return: An [N, 1] vector of densities computed as the sum of individual probabilities.
        """
        assert self.bandwidth is not None, (
            "The scoring of points requires a fitted kernel density estimator."
        )
        if not tf.is_tensor(query_points):
            query_points = tf.convert_to_tensor(query_points)
        if query_points.shape.rank == 1:
            query_points = tf.expand_dims(query_points, axis=-1)
        return (
            tf.expand_dims(
                tf.reduce_mean([kernel._prob(query_points) for kernel in self.kernels], axis=0), 
                axis=-1
            )
        )
