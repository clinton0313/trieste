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
Utilities for creating (Keras) neural network models to be used in the tests.
"""

from __future__ import annotations

from typing import Tuple, Callable

import tensorflow as tf

from trieste.data import Dataset
from trieste.models.keras import (
    DeepEnsemble,
    GaussianNetwork,
    KerasEnsemble,
    EpistemicUncertaintyNetwork,
    DirectEpistemicUncertaintyPredictor,
    get_tensor_spec_from_data,
)
from trieste.models.keras.architectures import EpistemicUncertaintyNetwork
from trieste.models.optimizer import KerasOptimizer


def trieste_keras_ensemble_model(
    example_data: Dataset,
    ensemble_size: int,
    independent_normal: bool = False,
) -> KerasEnsemble:

    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)

    networks = [
        GaussianNetwork(
            input_tensor_spec,
            output_tensor_spec,
            hidden_layer_args=[
                {"units": 32, "activation": "relu"},
                {"units": 32, "activation": "relu"},
            ],
            independent=independent_normal,
        )
        for _ in range(ensemble_size)
    ]
    keras_ensemble = KerasEnsemble(networks)

    return keras_ensemble


def trieste_deep_ensemble_model(
    example_data: Dataset,
    ensemble_size: int,
    bootstrap_data: bool = False,
    independent_normal: bool = False,
) -> Tuple[DeepEnsemble, KerasEnsemble, KerasOptimizer]:

    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, independent_normal)

    optimizer = tf.keras.optimizers.Adam()
    fit_args = {
        "batch_size": 32,
        "epochs": 10,
        "callbacks": [],
        "verbose": 0,
    }
    optimizer_wrapper = KerasOptimizer(optimizer, fit_args)

    model = DeepEnsemble(keras_ensemble, optimizer_wrapper, bootstrap_data)

    return model, keras_ensemble, optimizer_wrapper


def trieste_keras_epistemic_networks(
    data: Dataset,
    e_num_hidden_layers: int = 4,
    e_units: int = 128,
    e_activation: str = "relu",
    f_model_builder: Callable = trieste_keras_ensemble_model,
    **f_model_args
) -> tuple[DeepEnsemble, EpistemicUncertaintyNetwork]:

    f_model = f_model_builder(data, **f_model_args)

    e_input_tensor_spec, e_output_tensor_spec = get_tensor_spec_from_data(data)

    hidden_layer_args = []
    for _ in range(e_num_hidden_layers):
        hidden_layer_args.append(
            {
                "units": e_units, 
                "activation": e_activation
            }
        )

    e_model = EpistemicUncertaintyNetwork(
        e_input_tensor_spec,
        e_output_tensor_spec,
        hidden_layer_args
    )

    return f_model, e_model

def trieste_direct_epistemic_uncertainty_prediction(
    data: Dataset, 
    _init_buffer_iters: bool = 0
) -> DirectEpistemicUncertaintyPredictor:

    ensemble_params = {
        "ensemble_size": 5,
        "independent_normal": False
    }
    f_keras_ensemble, e_predictor = trieste_keras_epistemic_networks(
        data, 
        f_model_builder=trieste_keras_ensemble_model,
        **ensemble_params        
    )

    fit_args = {
        "batch_size": 16,
        "epochs": 1000,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100)
        ],
        "verbose": 0,
    }
    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.001), fit_args)

    f_ensemble = DeepEnsemble(f_keras_ensemble, optimizer)

    model = DirectEpistemicUncertaintyPredictor(
        model={"f_model": f_ensemble, "e_model": e_predictor},
        optimizer=optimizer, _init_buffer_iters=_init_buffer_iters
    )

    return model
