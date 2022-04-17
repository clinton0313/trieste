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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...types import TensorType
from typing import Callable, Union

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

def normal_inverse_gamma_negative_log_likelihood(
    y_true: TensorType, 
    y_pred: TensorType,
) -> TensorType:
    '''
    Computes the loss of the normal inverse gamma as computed by negative log likelihood estimation.

    :param y_true: The output variable values.
    :param y_pred: The four output parameters of the deep evidential regression model
        that characterize the normal inverse gamma distribution given in the order: gamma,
        lambda, alpha, beta.
    :return: The loss values
    '''

    gamma, v, alpha, beta = tf.split(y_pred, 4, axis=-1)

    omega = 2 * beta * (1 + v)

    negative_log_likelihood = (
        0.5 * tf.math.log(np.math.pi/v)
        - alpha * tf.math.log(omega)
        + (alpha + 0.5) * tf.math.log(v * (y_true - gamma) **2 + omega)
        + tf.math.lgamma(alpha)
        - tf.math.lgamma(alpha + 0.5)
    )
    return tf.reduce_mean(negative_log_likelihood, axis=0)


def normal_inverse_gamma_regularizer(
    y_true: TensorType, 
    y_pred: TensorType,
) -> TensorType:
    '''
    Computes the regularization loss for the Normal Inverse Gamma distribution for Deep
    Evidential Regression.
    
    :param y_true: The output variable values.
    :param y_pred: The four output parameters of the deep evidential regression model
        that characterize the normal inverse gamma distribution given in the order: gamma,
        lambda, alpha, beta.
    :return: The loss values
    '''
    gamma, v, alpha, _ = tf.split(y_pred, 4, axis=-1)
    return tf.reduce_mean(tf.abs(y_true - gamma) * (2*v + alpha), axis=0)

def deep_evidential_regression_loss(
    y_true: TensorType, 
    y_pred: TensorType, 
    coeff: float = 1e-2, 
) -> TensorType:
    '''
    Maximum likelihood objective for deep evidential regression model using negative 
    log likelihood or sum of squares loss of the normal inverse gamma distribution.

    :param y_true: The output variable values.
    :param y_pred: The four output parameters of the deep evidential regression model
        that characterize the normal inverse gamma distribution given in the order: gamma,
        lambda, alpha, beta.
    :param coeff: Regularization weight coefficient.
    :return: The model evidence values.  
    '''
    if y_true.shape.rank == 1:
        y_true = tf.expand_dims(y_true, axis=-1)
    if y_pred.shape.rank == 1:
        y_pred = tf.expand_dims(y_pred, axis=0)
    tf.debugging.assert_shapes([(y_pred, (y_pred.shape[0], 4))])

    loss = normal_inverse_gamma_negative_log_likelihood(y_true, y_pred)
    regularization = normal_inverse_gamma_regularizer(y_true, y_pred)

    return loss + coeff * regularization


def build_deep_evidential_regression_loss(
    coeff: float
) -> Callable:

    """
    Builds a custom loss function for Deep Evidential Regression model.

    :param coeff: Coefficient of the loss regularizer.
    :return: Loss function for the Deep Evidential Regression model. 
    """

    def loss_fn(y_true, y_pred):
        return deep_evidential_regression_loss(
            y_true,
            y_pred,
            coeff
        )
    return loss_fn

class DeepEvidentialCallback(tf.keras.callbacks.Callback):
    def __init__(self, reg_weight, maxi_rate, epsilon, verbose):
        self.reg_weight = reg_weight
        self.maxi_rate = maxi_rate
        self.epsilon = epsilon
        self.verbose = verbose

    def on_batch_end(self, epoch, logs=None):
        self.reg_weight.assign_add(self.maxi_rate * (logs["output_2_loss"] - self.epsilon))

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose == 1 and epoch % 100 == 0:
            print(f"Epoch: {epoch};  Loss = {logs['loss']:4f}; NLL_LOSS = {logs['output_1_loss']:4f}; reg_loss = {logs['output_2_loss']:4f}; lambda: {self.reg_weight.numpy():4f}")
