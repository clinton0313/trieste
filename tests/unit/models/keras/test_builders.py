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

from typing import Union

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import empty_dataset
from trieste.models.keras import build_vanilla_keras_ensemble, build_vanilla_deup
from trieste.models.keras.architectures import EpistemicUncertaintyNetwork


@pytest.mark.parametrize("units, activation", [(10, "relu"), (50, tf.keras.activations.tanh)])
@pytest.mark.parametrize("ensemble_size", [2, 5])
@pytest.mark.parametrize("independent_normal", [False, True])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
def test_build_vanilla_keras_ensemble(
    ensemble_size: int,
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    independent_normal: bool,
) -> None:
    example_data = empty_dataset([1], [1])
    keras_ensemble = build_vanilla_keras_ensemble(
        example_data,
        ensemble_size,
        num_hidden_layers,
        units,
        activation,
        independent_normal,
    )

    assert keras_ensemble.ensemble_size == ensemble_size
    assert len(keras_ensemble.model.layers) == num_hidden_layers * ensemble_size + 3 * ensemble_size
    if independent_normal:
        assert isinstance(keras_ensemble.model.layers[-1], tfp.layers.IndependentNormal)
    else:
        assert isinstance(keras_ensemble.model.layers[-1], tfp.layers.MultivariateNormalTriL)
    if num_hidden_layers > 0:
        for layer in keras_ensemble.model.layers[ensemble_size : -ensemble_size * 2]:
            assert layer.units == units
            assert layer.activation == activation or layer.activation.__name__ == activation

@pytest.mark.direct_epistemic
@pytest.mark.parametrize("ensemble_size", [3, 5])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3, 5])
@pytest.mark.parametrize("units", [5, 25, 50])
@pytest.mark.parametrize("e_num_hidden_layers", [1, 3, 5])
@pytest.mark.parametrize("e_units", [5, 25, 50])
@pytest.mark.parametrize("e_activation", ["relu", tf.keras.activations.tanh])
def test_build_vanilla_direct_epistemic_predictor(
    ensemble_size: int,
    num_hidden_layers: int,
    units: int,
    e_num_hidden_layers: int,
    e_units: int,
    e_activation: Union[str, tf.keras.layers.Activation]
) -> None:
    example_data = empty_dataset([1], [1])

    f_keras_ensemble, e_predictor = build_vanilla_deup(
        data=example_data, 
        f_model_builder=build_vanilla_keras_ensemble,
        ensemble_size=ensemble_size,
        num_hidden_layers=num_hidden_layers,
        units=units,
        activation="relu",
        e_num_hidden_layers=e_num_hidden_layers,
        e_units=e_units,
        e_activation=e_activation
    )

    assert f_keras_ensemble.ensemble_size == ensemble_size
    assert len(f_keras_ensemble.model.layers) == num_hidden_layers * ensemble_size + 3 * ensemble_size

    if num_hidden_layers > 0:
        for layer in f_keras_ensemble.model.layers[ensemble_size : -ensemble_size * 2]:
            assert layer.units == units

    assert isinstance(e_predictor, EpistemicUncertaintyNetwork)
    assert len(e_predictor.layers) == 2
    assert isinstance(e_predictor.layers[0], tf.keras.models.Sequential)
    assert isinstance(e_predictor.layers[-1], tf.keras.layers.Dense)
    assert len(e_predictor.layers[0].layers) == e_num_hidden_layers


    for layer in e_predictor.layers[0].layers:
        assert isinstance(layer, tf.keras.layers.Dense)
        assert layer.units == e_units
        assert layer.activation == e_activation or layer.activation.__name__ == e_activation