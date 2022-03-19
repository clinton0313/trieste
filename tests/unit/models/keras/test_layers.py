from typing import Any, Union

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from trieste.models.keras.layers import DropConnect


@pytest.fixture(name="x", params=[tf.constant([[5.0, 3.4, 2.6], [5.4, 3.2, 1.0]])])
def _x_fixture(request: Any) -> tf.Tensor:
    return request.param


@pytest.fixture(name="activation", params=[tf.nn.relu])
def _activation_fixture(request: Any):
    return request.param


@pytest.fixture(name="units", params=[3])
def _units_fixture(request: Any):
    return request.param


@pytest.fixture(name="layer", params=[DropConnect(units=3)])
def _layer_fixture(request: Any):
    return request.param


def test_dense_forward(
    layer: Dense, x: tf.Tensor, activation: Union[str, tf.keras.layers.Activation], units: int
) -> None:
    """Tests the forward method is working properly within the model without dropout"""

    layer.units = units
    layer.activation = activation

    inputs = Input(shape=(x.shape[-1],))
    outputs = layer(inputs)
    model = Model(inputs=inputs, outputs=outputs)

    dense_outputs = Dense(units=units, activation=activation, weights=model.get_weights())(inputs)
    dense_model = Model(inputs=inputs, outputs=dense_outputs)

    assert (
        (tf.equal(model(x), layer(x))).numpy().all()
    ), "Forward pass within a model is not a forward pass for the layer"
    assert (
        (tf.equal(model.predict(x), model(x))).numpy().all()
    ), "Model predict is not the same as a forward pass"
    assert (tf.equal(dense_model(x), model(x))).numpy().all(), "Forward pass calculations are wrong"


@pytest.mark.parametrize("rate", [0.0, (1 - 1e-12)])
def test_fit(
    layer: tf.keras.layers.Layer,
    x: tf.Tensor,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    rate: float,
) -> None:
    """Tests that the fit method with dropout is working properly"""
    y = tf.constant([[3.0, -4.0, 9.0], [5.0, 1.0, 6.0]])

    inputs = Input(shape=x.shape[-1])
    layer.activation = activation
    layer.rate = rate
    layer.units = units
    dense = Dense(units=units, activation=activation)

    drop_model = Model(inputs=inputs, outputs=layer(inputs))
    dense_model = Model(inputs=inputs, outputs=dense(inputs))
    drop_model.compile(Adam(), MeanAbsoluteError())
    dense_model.compile(Adam(), MeanAbsoluteError())

    bias = drop_model.get_weights()[1]
    weights = (
        drop_model.get_weights()[0]
        if rate == 0
        else tf.zeros(shape=drop_model.get_weights()[0].shape)
    )
    dense.set_weights([weights, bias])

    drop_fit = drop_model.fit(x, y)
    dense_fit = dense_model.fit(x, y)

    npt.assert_approx_equal(
        drop_fit.history["loss"][0],
        dense_fit.history["loss"][0],
        significant=3,
        err_msg=f"Expected {layer} to drop {rate} variables and get the same fit as a dense layer",
    )


@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("drop_layer", [DropConnect(units=1, use_bias=False)])
def test_dropout_rate(rate: float, drop_layer: tf.keras.layers.Layer) -> None:
    """Tests that weights are being dropped out at the write proportion"""
    drop_layer.rate = rate
    x1 = tf.constant([[1.0]])
    sims = 1000
    simulations = [np.sum(drop_layer(x1, training=True).numpy() == 0.0) for _ in range(sims)]

    # Test dropout up to twice the variance
    assert np.abs(np.sum(simulations) - rate * sims) <= 1.5 * rate * (1 - rate) * sims


@pytest.mark.parametrize("rate", [1.5, -1.0])
def test_dropout_rate_raises_value_error(rate: float, units: int) -> None:
    """Tests that value error is raised when given wrong probability rates"""
    with pytest.raises(ValueError):
        _ = DropConnect(rate=rate, units=units)
