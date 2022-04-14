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

from typing import Any, List

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import ShapeLike, empty_dataset, random_seed
from tests.util.models.keras.models import (
    trieste_mcdropout_model, trieste_dropout_network_model
)
from tests.util.models.models import fnc_2sin_x_over_3
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.keras import (
    MonteCarloDropout,
    DropoutNetwork,
    negative_log_likelihood,
)
from trieste.models.keras.architectures import DropConnectNetwork
from trieste.models.optimizer import KerasOptimizer, TrainingData

def _get_example_data(query_point_shape: ShapeLike) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), dtype=tf.float64)
    obs = fnc_2sin_x_over_3(qp)
    return Dataset(qp, obs)

def _get_linear_data(query_point_shape: ShapeLike, slope: str) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), minval=-4, maxval=4, dtype=tf.float32)
    if slope == "pos":
        obs = tf.multiply(3, qp)
    elif slope == "neg":
        obs = tf.multiply(-3, qp)
    return Dataset(qp, obs)

@pytest.fixture(name="rate", params=[0.1])
def _rate_fixture(request: Any) -> float:
    return request.param

@pytest.fixture(name="loss")
def _loss_fixture(request: Any) -> float:
    return "mse"

@pytest.mark.mcdropout
@pytest.mark.parametrize("optimizer", [tf.optimizers.Adam(), tf.optimizers.RMSprop()])
def test_mcdropout_repr(
    optimizer: tf.optimizers.Optimizer,
    rate: List,
    loss: str,
) -> None:
    example_data = empty_dataset([1], [1])
    dropout_network = trieste_dropout_network_model(example_data, rate)
    dropout_network.compile(optimizer, loss=loss)
    optimizer_wrapper = KerasOptimizer(optimizer, loss=loss)
    model = MonteCarloDropout(dropout_network, optimizer_wrapper)

    expected_repr = (
        f"MonteCarloDropout({dropout_network!r}, {optimizer_wrapper!r})"
    )

    assert type(model).__name__ in repr(model)
    assert repr(model) == expected_repr

def test_dropout_network_model_attributes(rate: List) -> None:
    example_data = empty_dataset([1], [1])
    model, dropout_network, optimizer = trieste_mcdropout_model(
        example_data, rate=rate
    )

    dropout_network.compile(optimizer=optimizer.optimizer, loss=optimizer.loss)

    assert model.model is dropout_network


@pytest.mark.mcdropout
def test_dropout_network_default_optimizer_is_correct(rate: List, loss: str) -> None:
    example_data = empty_dataset([1], [1])

    dropout_network = trieste_dropout_network_model(example_data, rate)
    model = MonteCarloDropout(dropout_network)
    default_loss = loss
    default_fit_args = {
        "batch_size": 32,
        "epochs": 1000,
        "verbose": 0,
    }
    del model.optimizer.fit_args["callbacks"]

    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args
    assert model.optimizer.loss == default_loss


@pytest.mark.mcdropout
def test_mcdropout_optimizer_changed_correctly(rate: List) -> None:
    example_data = empty_dataset([1], [1])

    custom_fit_args = {
        "verbose": 1,
        "epochs": 10,
        "batch_size": 10,
    }
    custom_optimizer = tf.optimizers.RMSprop()
    custom_loss = negative_log_likelihood
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args, custom_loss)

    dropout_network = trieste_dropout_network_model(example_data, rate)
    model = MonteCarloDropout(dropout_network, optimizer_wrapper)

    assert model.optimizer == optimizer_wrapper
    assert model.optimizer.optimizer == custom_optimizer
    assert model.optimizer.fit_args == custom_fit_args


@pytest.mark.mcdropout
def test_mcdropout_is_compiled(rate: List) -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_mcdropout_model(example_data, rate)

    assert model.model.compiled_loss is not None
    assert model.model.compiled_metrics is not None
    assert model.model.optimizer is not None


@pytest.mark.skip
def test_config_builds_mcdropout_and_default_optimizer_is_correct(rate: List) -> None:
    example_data = empty_dataset([1], [1])

    dropout_network = trieste_dropout_network_model(example_data, rate)

    model_config = {"model": dropout_network}
    model = create_model(model_config)
    default_fit_args = {
        "verbose": 0,
        "epochs": 100,
        "batch_size": 100,
    }

    assert isinstance(model, MonteCarloDropout)
    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.keras.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args


@pytest.mark.mcdropout
@pytest.mark.parametrize("dataset_size", [10, 100])
def test_mcdropout_predict_call_shape(dataset_size: int, rate: List) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_mcdropout_model(example_data, rate)

    predicted_means, predicted_vars = model.predict(example_data.query_points)

    assert tf.is_tensor(predicted_vars)
    assert predicted_vars.shape == example_data.observations.shape
    assert tf.is_tensor(predicted_means)
    assert predicted_means.shape == example_data.observations.shape


@pytest.mark.mcdropout
@pytest.mark.parametrize("num_samples", [6, 12])
@pytest.mark.parametrize("dataset_size", [4, 8])
def test_mcdropout_sample_call_shape(num_samples: int, dataset_size: int, rate: List) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_mcdropout_model(example_data, rate)

    samples = model.sample(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples, dataset_size, 1]


@random_seed
@pytest.mark.mcdropout
def test_mcdropout_optimize_with_defaults(rate: List) -> None:
    example_data = _get_example_data([100, 1])

    dropout_network = trieste_dropout_network_model(example_data, rate)

    model = MonteCarloDropout(dropout_network)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]

@pytest.mark.mcdropout
def test_mcdropout_learning_rate_resets(rate: List) -> None:
    example_data = _get_example_data([100,1])

    dropout_network = trieste_dropout_network_model(example_data, rate)

    model = MonteCarloDropout(dropout_network)

    model.optimize(example_data)
    lr1 = model.model.history.history["lr"]

    model.optimize(example_data)
    lr2 = model.model.history.history["lr"]

    assert lr1[0] == lr2[0]


@random_seed
@pytest.mark.mcdropout
def test_mcdropout_optimizer_learns_new_data(rate: List) -> None:
    
    positive_slope = _get_linear_data([20,1], "pos")
    negative_slope = _get_linear_data([20,1], "neg")
    new_data = positive_slope + negative_slope
    qp = tf.constant([[1.]])

    dropout_network = trieste_dropout_network_model(positive_slope, rate, DropoutNetwork)

    model = MonteCarloDropout(dropout_network)

    model.optimize(positive_slope)
    pred1, _ = model.predict(qp)
    model.optimize(new_data)
    pred2, _ = model.predict(qp)

    assert np.abs(pred1-pred2) > 1


@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("epochs", [5, 15])
@pytest.mark.parametrize("learning_rate", [0.1, 0.01])
@pytest.mark.parametrize("rate", [0.1, 0.2])
def test_mcdropout_optimize(rate: float, epochs: int, learning_rate: float) -> None:
    example_data = _get_example_data([20, 1])

    dropout_network = trieste_dropout_network_model(example_data, rate, DropoutNetwork)

    custom_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    custom_fit_args = {
        "verbose": 0,
        "epochs": epochs,
        "batch_size": 10,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=80, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.3, patience=15
            )
        ]
    }
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args)

    model = MonteCarloDropout(dropout_network, optimizer=optimizer_wrapper, learning_rate=learning_rate)

    model.optimize(example_data)

    lr_hist = model.model.history.history["lr"]
    loss = model.model.history.history["loss"]
    # breakpoint()
    assert loss[-1] < loss[0]
    assert len(loss) == epochs
    npt.assert_almost_equal(lr_hist[0], learning_rate, decimal=3)

@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("layer", [DropoutNetwork, DropConnectNetwork])
def test_mcdropout_loss_with_different_layers_and_reset_lr(
    rate: List, 
    loss: str,
    layer
) -> None:
    example_data_1 = _get_example_data([200, 1])
    example_data_2 = _get_example_data([200, 1])

    model_1 = MonteCarloDropout(
        trieste_dropout_network_model(example_data_1, rate, layer),
        KerasOptimizer(tf.optimizers.Adam(), loss=loss),
        learning_rate=0.01
    )
    model_1.optimize(example_data_1)

    model_2 = MonteCarloDropout(
        trieste_dropout_network_model(example_data_2, rate, layer),
        KerasOptimizer(tf.optimizers.Adam(), loss=loss),
        learning_rate=0.01
    )
    model_2.optimize(example_data_2)

    loss_1 = model_1.model.evaluate(example_data_1.astuple()[0], example_data_1.astuple()[1])
    loss_2 = model_2.model.evaluate(example_data_2.astuple()[0], example_data_2.astuple()[1])
    
    npt.assert_almost_equal(loss_1, loss_2, decimal=3)


@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("num_samples", [50, 100, 200])
@pytest.mark.parametrize("num_passes", [50, 100, 200])
def test_mcdropout_predict_num_passes(rate: List, loss: str, num_samples, num_passes) -> None:
    example_data = _get_example_data([100, 1])
    transformed_x, transformed_y = example_data.astuple()

    model = MonteCarloDropout(
        trieste_dropout_network_model(example_data, rate, DropoutNetwork),
        KerasOptimizer(tf.optimizers.Adam(), loss=loss),
        learning_rate=0.01,
        num_passes=num_passes
    )
    model.optimize(example_data)

    reference_model = trieste_dropout_network_model(example_data, rate, DropoutNetwork)
    reference_model.compile(optimizer=tf.optimizers.Adam(), loss=loss)
    reference_model.fit(transformed_x, transformed_y)
    reference_model.set_weights(model.model.get_weights())

    sample_means = tf.reduce_mean(model.sample(example_data.query_points, num_samples=num_samples), axis=0)
    predicted_means, _ = model.predict(example_data.query_points)
    reference_means = reference_model(example_data.query_points)
    
    npt.assert_allclose(predicted_means, sample_means, atol=0.1)
    npt.assert_allclose(predicted_means, reference_means, atol=0.1)


@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("num_samples", [1000, 5000, 10000])
def test_mcdropout_sample(rate: List, num_samples) -> None:
    example_data = _get_example_data([100, 1])
    model, _, _ = trieste_mcdropout_model(example_data, rate)

    samples = model.sample(example_data.query_points, num_samples)
    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2, axis=0)

    ref_mean, ref_variance = model.predict(example_data.query_points)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))

    npt.assert_allclose(sample_mean, ref_mean, atol=4 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=8 * error)
