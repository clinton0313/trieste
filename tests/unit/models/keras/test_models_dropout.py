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

from typing import Any

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
    MCDropout,
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

@pytest.mark.mcdropout
@pytest.mark.parametrize("optimizer", [tf.optimizers.Adam(), tf.optimizers.RMSprop()])
def test_mcdropout_repr(
    optimizer: tf.optimizers.Optimizer,
    bootstrap_data: bool,
) -> None:
    example_data = empty_dataset([1], [1])

    dropout_network = trieste_dropout_network_model(example_data, _rate_fixture)
    dropout_network.compile(optimizer, loss="mse")
    optimizer_wrapper = KerasOptimizer(optimizer, loss="mse")
    model = MCDropout(dropout_network, optimizer_wrapper)

    expected_repr = (
        f"MCDropout({dropout_network!r}, {optimizer_wrapper!r})"
    )

    assert type(model).__name__ in repr(model)
    assert repr(model) == expected_repr

def test_dropout_network_model_attributes() -> None:
    example_data = empty_dataset([1], [1])
    model, dropout_network, optimizer = trieste_mcdropout_model(
        example_data, rate=_rate_fixture
    )

    dropout_network.compile(optimizer=optimizer.optimizer, loss=optimizer.loss)

    assert model.model is dropout_network


@pytest.mark.mcdropout
def test_dropout_network_default_optimizer_is_correct() -> None:
    example_data = empty_dataset([1], [1])

    dropout_network = trieste_dropout_network_model(example_data, _rate_fixture)
    model = MCDropout(dropout_network)
    default_loss = "mse"
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
def test_mcdropout_optimizer_changed_correctly() -> None:
    example_data = empty_dataset([1], [1])

    custom_fit_args = {
        "verbose": 1,
        "epochs": 10,
        "batch_size": 10,
    }
    custom_optimizer = tf.optimizers.RMSprop()
    custom_loss = negative_log_likelihood
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args, custom_loss)

    dropout_network = trieste_dropout_network_model(example_data, _rate_fixture)
    model = MCDropout(dropout_network, optimizer_wrapper)

    assert model.optimizer == optimizer_wrapper
    assert model.optimizer.optimizer == custom_optimizer
    assert model.optimizer.fit_args == custom_fit_args


@pytest.mark.mcdropout
def test_mcdropout_is_compiled() -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_mcdropout_model(example_data, _rate_fixture)

    assert model.model.compiled_loss is not None
    assert model.model.compiled_metrics is not None
    assert model.model.optimizer is not None


@pytest.mark.skip
def test_config_builds_mcdropout_and_default_optimizer_is_correct() -> None:
    example_data = empty_dataset([1], [1])

    dropout_network = trieste_dropout_network_model(example_data, _rate_fixture)

    model_config = {"model": dropout_network}
    model = create_model(model_config)
    default_fit_args = {
        "verbose": 0,
        "epochs": 100,
        "batch_size": 100,
    }

    assert isinstance(model, MCDropout)
    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.keras.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args


@pytest.mark.mcdropout
@pytest.mark.parametrize("dataset_size", [10, 100])
def test_mcdropout_predict_call_shape(dataset_size: int) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_mcdropout_model(example_data, _rate_fixture)

    predicted_means, predicted_vars = model.predict(example_data.query_points)

    assert tf.is_tensor(predicted_vars)
    assert predicted_vars.shape == example_data.observations.shape
    assert tf.is_tensor(predicted_means)
    assert predicted_means.shape == example_data.observations.shape


@pytest.mark.mcdropout
@pytest.mark.parametrize("num_samples", [6, 12])
@pytest.mark.parametrize("dataset_size", [4, 8])
def test_mcdropout_sample_call_shape(num_samples: int, dataset_size: int) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_mcdropout_model(example_data, _rate_fixture)

    samples = model.sample(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples, dataset_size, 1]


@random_seed
@pytest.mark.mcdropout
def test_mcdropout_optimize_with_defaults() -> None:
    example_data = _get_example_data([100, 1])

    dropout_network = trieste_dropout_network_model(example_data, _rate_fixture)

    model = MCDropout(dropout_network)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]

@pytest.mark.mcdropout
def test_mcdropout_learning_rate_resets() -> None:
    example_data = _get_example_data([100,1])

    dropout_network = trieste_dropout_network_model(example_data, _rate_fixture)

    model = MCDropout(dropout_network)

    model.optimize(example_data)
    lr1 = model.model.history.history["lr"]

    model.optimize(example_data)
    lr2 = model.model.history.history["lr"]

    assert lr1[0] == lr2[0]


@random_seed
@pytest.mark.mcdropout
def test_mcdropout_optimizer_learns_new_data() -> None:
    
    positive_slope = _get_linear_data([20,1], "pos")
    negative_slope = _get_linear_data([20,1], "neg")
    new_data = positive_slope + negative_slope
    qp = tf.constant([[1.]])

    dropout_network = trieste_dropout_network_model(positive_slope, _rate_fixture, DropoutNetwork)

    model = MCDropout(dropout_network)

    model.optimize(positive_slope)
    pred1, _ = model.predict(qp)
    model.optimize(new_data)
    pred2, _ = model.predict(qp)

    assert np.abs(pred1-pred2) > 1


@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("epochs", [5, 15])
@pytest.mark.parametrize("learning_rate", [0.01, 0.1])
@pytest.mark.parametrize("rate", [0.1, 0.2])
def test_mcdropout_optimize(rate: float, epochs: int, learning_rate: float) -> None:
    example_data = _get_example_data([20, 1])

    dropout_network = trieste_dropout_network_model(example_data, rate, DropoutNetwork)

    custom_optimizer = tf.optimizers.RMSprop()
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

    model = MCDropout(dropout_network, optimizer=optimizer_wrapper, learning_rate=learning_rate)

    model.optimize(example_data)

    lr_hist = model.model.history.history["lr"]
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]
    assert len(loss) == epochs
    npt.assert_almost_equal(lr_hist[0], learning_rate, decimal=3)