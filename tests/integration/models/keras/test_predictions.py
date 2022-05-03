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

import pytest
import tensorflow as tf

from tests.util.misc import branin_dataset, random_seed
from trieste.models.keras import (
    DeepEnsemble,
    DropoutNetwork,
    DropConnectNetwork,
    MonteCarloDropout,
    build_vanilla_keras_ensemble,
    build_vanilla_keras_mcdropout,
)
from trieste.models.optimizer import KerasOptimizer

@pytest.mark.slow
@random_seed
def test_neural_network_ensemble_predictions_close_to_actuals(keras_float: None) -> None:
    ensemble_size = 5
    dataset_size = 1000

    example_data = branin_dataset(dataset_size)

    keras_ensemble = build_vanilla_keras_ensemble(example_data, ensemble_size, 2, 50)
    optimizer = tf.keras.optimizers.Adam()
    fit_args = {
        "batch_size": 20,
        "epochs": 1000,
        "callbacks": [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
        "verbose": 0,
    }
    model = DeepEnsemble(
        keras_ensemble,
        KerasOptimizer(optimizer, fit_args),
        False,
    )
    model.optimize(example_data)

    predicted_means, _ = model.predict(example_data.query_points)
    mean_abs_deviation = tf.reduce_mean(tf.abs(predicted_means - example_data.observations))

    # somewhat arbitrary accuracy level, seems good for the range of branin values
    assert mean_abs_deviation < 2


@pytest.mark.slow
@pytest.mark.parametrize(
    "dropout_network, max_error, rate", 
    [(DropoutNetwork, 10., 0.03), (DropConnectNetwork, 3., 0.05)]
    )
@random_seed
def test_dropout_network_predictions_close_to_actuals(
    dropout_network: DropoutNetwork, 
    max_error: float,
    rate: float
) -> None:

    example_data = branin_dataset(1000)

    dropout_nn = build_vanilla_keras_mcdropout(
        data=example_data, 
        rate=rate,
        dropout_network=dropout_network
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    fit_args = {
        "batch_size": 32,
        "epochs": 1000,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=80), 
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.3, patience=15)
            ],
        "verbose": 0
    }

    model = MonteCarloDropout(
        model=dropout_nn,
        optimizer=KerasOptimizer(optimizer, fit_args),
        num_passes=200
    )
    model.optimize(example_data)

    predicted_means, _ = model.predict(example_data.query_points)
    mean_abs_deviation = tf.reduce_mean(tf.abs(predicted_means - example_data.observations))

    # Abitrary accuracy levels given the dropout rates that hinder overfitting to the data
    assert mean_abs_deviation < max_error
