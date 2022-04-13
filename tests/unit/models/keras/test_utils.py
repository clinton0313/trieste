# Copyright 2021 The Bellman Contributors
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


from distutils.command.build import build
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import ShapeLike, empty_dataset, random_seed
from trieste.data import Dataset
from trieste.models.keras.utils import (
    build_deep_evidential_regression_loss,
    deep_evidential_regression_loss,
    get_tensor_spec_from_data,
    normal_inverse_gamma_negative_log_likelihood,
    normal_inverse_gamma_regularizer,
    normal_inverse_gamma_sum_of_squares, 
    sample_with_replacement,

)
from trieste.types import TensorType
from typing import Callable, Union


def test_get_tensor_spec_from_data_raises_for_incorrect_dataset() -> None:

    dataset = empty_dataset([1], [1])

    with pytest.raises(ValueError):
        get_tensor_spec_from_data(dataset.query_points)


@pytest.mark.parametrize(
    "query_point_shape, observation_shape",
    [([1], [1]), ([2], [1]), ([5], [1]), ([5], [2]), ([3, 2], [3, 1])],
)
def test_get_tensor_spec_from_data(
    query_point_shape: ShapeLike, observation_shape: ShapeLike
) -> None:
    dataset = empty_dataset(query_point_shape, observation_shape)
    input_spec, output_spec = get_tensor_spec_from_data(dataset)

    assert input_spec.shape == query_point_shape
    assert input_spec.dtype == dataset.query_points.dtype
    assert input_spec.name == "query_points"

    assert output_spec.shape == observation_shape
    assert output_spec.dtype == dataset.observations.dtype
    assert output_spec.name == "observations"


def test_sample_with_replacement_raises_for_invalid_dataset() -> None:

    dataset = empty_dataset([1], [1])

    with pytest.raises(ValueError):
        sample_with_replacement(dataset.query_points)


def test_sample_with_replacement_raises_for_empty_dataset() -> None:

    dataset = empty_dataset([1], [1])

    with pytest.raises(tf.errors.InvalidArgumentError):
        sample_with_replacement(dataset)


@random_seed
@pytest.mark.parametrize("rank", [2, 3])
def test_sample_with_replacement_seems_correct(rank: int) -> None:

    n_rows = 100
    if rank == 2:
        x = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
        y = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
    elif rank == 3:
        x = tf.constant(np.arange(0, n_rows, 1).repeat(2), shape=[n_rows, 2, 1])
        y = tf.constant(np.arange(0, n_rows, 1).repeat(2), shape=[n_rows, 2, 1])
    dataset = Dataset(x, y)

    dataset_resampled = sample_with_replacement(dataset)

    # basic check that original dataset has not been changed
    assert tf.reduce_all(dataset.query_points == x)
    assert tf.reduce_all(dataset.observations == y)

    # x and y should be resampled the same, and should differ from the original
    assert tf.reduce_all(dataset_resampled.query_points == dataset_resampled.observations)
    assert tf.reduce_any(dataset_resampled.query_points != x)
    assert tf.reduce_any(dataset_resampled.observations != y)

    # values are likely to repeat due to replacement
    _, _, count = tf.unique_with_counts(tf.squeeze(dataset_resampled.query_points[:, 0]))
    assert tf.reduce_any(count > 1)

    # mean of bootstrap samples should be close to true mean
    mean = [
        tf.reduce_mean(
            tf.cast(sample_with_replacement(dataset).query_points[:, 0], dtype=tf.float32)
        )
        for _ in range(100)
    ]
    x = tf.cast(x[:, 0], dtype=tf.float32)
    assert (tf.reduce_mean(mean) - tf.reduce_mean(x)) < 1
    assert tf.math.abs(tf.math.reduce_std(mean) - tf.math.reduce_std(x) / 10.0) < 0.1


@pytest.mark.parametrize(
    "y_true, gamma, lamb, alpha, beta, true_loss",
    [
        (1., 1., 0.5, 0.4, 0.3, 0.86625844), #With gamma function as paper: 1.5965598
        (1.8, 2., 0.3, 0.1, 0.7, 1.4777126) #With gamma function as paper: 3.3321915
    ]
)
def test_normal_inverse_gamma_negative_log_likelihood_is_accurate(
    y_true: float,
    gamma: float,
    lamb: float,
    alpha: float,
    beta: float,
    true_loss: float
) -> None:
    loss = normal_inverse_gamma_negative_log_likelihood(y_true, gamma, lamb, alpha, beta)
    npt.assert_approx_equal(loss, true_loss)

@pytest.mark.parametrize(
    "y_true, gamma, lamb, alpha, beta, true_loss",
    [
        (1., 1., 0.5, 0.4, 0.3, 0.4054652),
        (1.8, 2., 0.3, 0.1, 0.7, 1.203083)
    ]
)
def test_normal_inverse_gamma_sum_of_squares_is_accurate(
    y_true: float,
    gamma: float,
    lamb: float,
    alpha: float,
    beta: float,
    true_loss: float
) -> None:
    loss = normal_inverse_gamma_sum_of_squares(y_true, gamma, lamb, alpha, beta)
    npt.assert_approx_equal(loss, true_loss)

@pytest.mark.parametrize(
    "y_true, gamma, lamb, alpha, true_loss",
    [
        (3., 1., 0.5, 0.4, 2.6),
        (1.8, 2., 0.3, 0.1, 0.1)
    ]
)
def test_normal_inverse_gamma_regularizer_is_accurate(
    y_true: float,
    gamma: float,
    lamb: float,
    alpha: float,
    true_loss: float
) -> None:
    loss = normal_inverse_gamma_regularizer(y_true, gamma, lamb, alpha)
    npt.assert_approx_equal(loss, true_loss)

@pytest.mark.parametrize(
    "y_true, y_pred, use_sos, true_loss",
    [
        (
            tf.constant([[1.], [1.8]]),
            tf.constant([
                [1., 0.5, 0.4, 0.3],
                [2., 0.3, 0.1, 0.7]
            ]),
            False,
            1.2219855 #If using the gamma functions from paper 2.4643755
        ),
        (
            tf.constant([[1.], [1.8]]),
            tf.constant([
                [1., 0.5, 0.4, 0.3],
                [2., 0.3, 0.1, 0.7]
            ]),
            True,
            0.85427415
        )
    ]
)
def test_deep_evidential_regression_loss_is_accurate(
    y_true: TensorType,
    y_pred: TensorType,
    use_sos: bool,
    true_loss: float
) -> None:
    if use_sos:
        loss = deep_evidential_regression_loss(y_true, y_pred)
    else:
        loss = deep_evidential_regression_loss(
            y_true, 
            y_pred, 
            loss_fn = normal_inverse_gamma_negative_log_likelihood
        )
    npt.assert_approx_equal(loss, true_loss)

@pytest.mark.parametrize(
    "y_pred",
    [
        tf.zeros((10, 3)),
        tf.ones((300,))
    ]
)
def test_deep_evidential_regression_loss_asserts_shape(
    y_pred: TensorType,
) -> None:
    y_true = tf.zeros((y_pred.shape[0],))
    with pytest.raises(ValueError):
        deep_evidential_regression_loss(y_true, y_pred)


@pytest.mark.deep_evidential
@pytest.mark.parametrize(
    "loss_fn",
    [
        "SOS",
        normal_inverse_gamma_negative_log_likelihood
    ]
)
@pytest.mark.parametrize("coeff", [0.5, 1.5])
def test_build_deep_evidential_regression_loss(
    loss_fn: Union[str, Callable],
    coeff: float
) -> None:

    y_pred = tf.constant([[2., 1., 1.5, 2.]])
    y_true = tf.constant([[1.]])

    gamma, lamb, alpha, beta = tf.split(y_pred, 4, axis=-1)

    if isinstance(loss_fn, str):
        if loss_fn == "SOS":
            base_loss = normal_inverse_gamma_sum_of_squares
        elif loss_fn == "NLL":
            base_loss = normal_inverse_gamma_negative_log_likelihood
    else:
        base_loss = loss_fn
    
    reference_loss = base_loss(y_true, gamma, lamb, alpha, beta) \
                    + coeff * normal_inverse_gamma_regularizer(y_true, gamma, lamb, alpha)
    
    loss = build_deep_evidential_regression_loss(base_loss, coeff)
    built_loss = loss(y_true, y_pred)

    npt.assert_almost_equal(built_loss, reference_loss)


@pytest.mark.deep_evidential
def build_deep_evidential_regression_loss_raises_value_error() -> None:
    with pytest.raises(ValueError):
        build_deep_evidential_regression_loss("test")


@pytest.mark.deep_evidential
def build_deep_evidential_regression_loss_has_name() -> None:
    '''Tensorflow requires that loss function has a __name__ attribute.'''
    loss = build_deep_evidential_regression_loss()
    assert loss.__name__ == "loss"