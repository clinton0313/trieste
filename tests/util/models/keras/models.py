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
from distutils.command.build import build

from typing import Tuple, Optional

import tensorflow as tf

from trieste.data import Dataset
from trieste.models.keras import (
    DeepEnsemble,
    DeepEvidentialRegression,
    GaussianNetwork,
    KerasEnsemble,
<<<<<<< HEAD
    DropoutNetwork,
    DropConnectNetwork,
    MonteCarloDropout,
=======
    build_vanilla_keras_deep_evidential,
>>>>>>> clinton/der_model
    get_tensor_spec_from_data,
)
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

<<<<<<< HEAD
def trieste_dropout_network_model(
    example_data: Dataset,
    rate: float = 0.1,
    dropout: DropoutNetwork = DropoutNetwork
) -> DropoutNetwork:
    
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)

    dropout_network = dropout(
        input_tensor_spec,
        output_tensor_spec,
        hidden_layer_args=[
            {"units": 300, "activation": "relu"},
            {"units": 300, "activation": "relu"},
            {"units": 300, "activation": "relu"},
        ],
        rate=rate
    )

    return dropout_network

def trieste_mcdropout_model(
    example_data:Dataset, 
    rate: float = 0.1,
    dropout:DropoutNetwork=DropoutNetwork
) -> MonteCarloDropout:

    dropout_network = trieste_dropout_network_model(
        example_data, 
        rate=rate,
        dropout=dropout
    )

    optimizer = tf.keras.optimizers.Adam(0.01)
    fit_args = {
        "batch_size": 10,
        "epochs": 100,
        "verbose": 0,
    }
    optimizer_wrapper = KerasOptimizer(optimizer, fit_args)

    model = MonteCarloDropout(dropout_network, optimizer_wrapper)

    return model, dropout_network, optimizer_wrapper

class MCDropConnect(MonteCarloDropout):
    '''Placeholder class for Bayesian optimization integration tests.'''
    def __init__(self, model:DropConnectNetwork, **model_args):
        super().__init__(model=model, **model_args)
=======

def trieste_deep_evidential_model(
    example_data: Dataset,
    optimizer: Optional[KerasOptimizer] = None,
    **model_args
) -> DeepEvidentialRegression:

    evidential_network = build_vanilla_keras_deep_evidential(example_data)
    model = DeepEvidentialRegression(evidential_network, optimizer, **model_args)

    return model
>>>>>>> clinton/der_model
