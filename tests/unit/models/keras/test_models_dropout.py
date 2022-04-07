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
    trieste_deep_ensemble_model, trieste_keras_ensemble_model, 
    trieste_mcdropout_model, trieste_dropout_network_model
)
from tests.util.models.models import fnc_2sin_x_over_3
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.keras import (
    MCDropout,
    DropoutNetwork,
    DeepEnsemble,
    KerasEnsemble,
    negative_log_likelihood,
    sample_with_replacement,
)
from trieste.models.optimizer import KerasOptimizer, TrainingData

@pytest.fixture(name="ensemble_size", params=[2, 5])
def _ensemble_size_fixture(request: Any) -> int:
    return request.param
    

@pytest.fixture(name="independent_normal", params=[False, True])
def _independent_normal_fixture(request: Any) -> bool:
    return request.param


@pytest.fixture(name="bootstrap_data", params=[False, True])
def _bootstrap_data_fixture(request: Any) -> bool:
    return request.param

def _get_example_data(query_point_shape: ShapeLike) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), dtype=tf.float64)
    obs = fnc_2sin_x_over_3(qp)
    return Dataset(qp, obs)


def _ensemblise_data(
    model: KerasEnsemble, data: Dataset, ensemble_size: int, bootstrap: bool = False
) -> TrainingData:
    inputs = {}
    outputs = {}
    for index in range(ensemble_size):
        if bootstrap:
            resampled_data = sample_with_replacement(data)
        else:
            resampled_data = data
        input_name = model.model.input_names[index]
        output_name = model.model.output_names[index]
        inputs[input_name], outputs[output_name] = resampled_data.astuple()

    return inputs, outputs

_ENSEMBLE_SIZE = 3
_RATE = 0.1

@pytest.mark.parametrize("optimizer", [tf.optimizers.Adam(), tf.optimizers.RMSprop()])
def test_mcdropout_repr(
    optimizer: tf.optimizers.Optimizer,
    bootstrap_data: bool,
) -> None:
    example_data = empty_dataset([1], [1])

    dropout_nn = trieste_dropout_network_model(example_data, _RATE)
    dropout_nn.compile(optimizer, loss="mse")
    optimizer_wrapper = KerasOptimizer(optimizer, loss="mse")
    model = MCDropout(dropout_nn, optimizer_wrapper)

    expected_repr = (
        f"MCDropout({dropout_nn!r}, {optimizer_wrapper!r})"
    )

    assert type(model).__name__ in repr(model)
    assert repr(model) == expected_repr

def test_dropout_network_model_attributes() -> None:
    example_data = empty_dataset([1], [1])
    model, dropout_nn, optimizer = trieste_mcdropout_model(
        example_data, rate=_RATE
    )

    dropout_nn.compile(optimizer=optimizer.optimizer, loss=optimizer.loss)

    assert model.model is dropout_nn

def test_dropout_network_default_optimizer_is_correct() -> None:
    example_data = empty_dataset([1], [1])
    # breakpoint()
    dropout_nn = trieste_dropout_network_model(example_data, _RATE)
    model = MCDropout(dropout_nn)
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

    dropout_nn = trieste_dropout_network_model(example_data, _RATE)
    model = MCDropout(dropout_nn, optimizer_wrapper)

    assert model.optimizer == optimizer_wrapper
    assert model.optimizer.optimizer == custom_optimizer
    assert model.optimizer.fit_args == custom_fit_args

def test_mcdropout_is_compiled() -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_mcdropout_model(example_data, _RATE)

    assert model.model.compiled_loss is not None
    assert model.model.compiled_metrics is not None
    assert model.model.optimizer is not None