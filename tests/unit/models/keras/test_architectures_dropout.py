from typing import Any, List, Union

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from tests.util.misc import empty_dataset
from trieste.models.keras import DropConnectNetwork, DropoutNetwork, get_tensor_spec_from_data
from trieste.models.keras.layers import DropConnect


@pytest.fixture(name="dropout_network", params=[DropoutNetwork, DropConnectNetwork])
def _dropout_network_fixture(request: Any) -> DropoutNetwork:
    return request.param


@pytest.fixture(name="query_point_shape", params=[[1], [5]])
def _query_point_shape_fixture(request: Any) -> List[int]:
    return request.param


@pytest.fixture(name="observation_shape", params=[[1], [2]])
def _observation_shape_fixture(request: Any) -> List[int]:
    return request.param


@pytest.mark.parametrize("num_hidden_layers, rate", [(1, 0.3), (3, 0.7), (5, 0.9)])
@pytest.mark.parametrize("units", [10, 50])
@pytest.mark.parametrize("activation", ["relu", tf.keras.activations.tanh])
def test_dropout_network_build_seems_correct(
    dropout_network: DropoutNetwork,
    query_point_shape: List[int],
    observation_shape: List[int],
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    rate: float,
) -> None:
    """Tests the correct consturction of dropout network architectures"""

    example_data = empty_dataset(query_point_shape, observation_shape)
    inputs, outputs = get_tensor_spec_from_data(example_data)
    hidden_layer_args = [
        {"units": units, "activation": activation} for _ in range(num_hidden_layers)
    ]

    dropout_nn = dropout_network(inputs, outputs, hidden_layer_args, rate)

    if not isinstance(rate, list):
        rate = [rate for _ in range(num_hidden_layers + 1)]

    # basics
    assert isinstance(dropout_nn, tf.keras.Model)

    # check the model has not been compiled
    assert dropout_nn.compiled_loss is None
    assert dropout_nn.compiled_metrics is None
    assert dropout_nn.optimizer is None

    # check correct number of layers and proerply constructed
    assert len(dropout_nn.layers) == 2

    if isinstance(dropout_nn, DropConnectNetwork):
        assert len(dropout_nn.layers[0].layers) == num_hidden_layers

        for layer in dropout_nn.layers[0].layers:
            assert isinstance(layer, DropConnect)
            assert layer.units == units
            assert layer.activation == activation or layer.activation.__name__ == activation

        assert isinstance(dropout_nn.layers[-1], DropConnect)
        assert dropout_nn.layers[-1].units == int(np.prod(outputs.shape))
        assert dropout_nn.layers[-1].activation == tf.keras.activations.linear

    elif isinstance(dropout_nn, DropoutNetwork):
        assert len(dropout_nn.layers[0].layers) == num_hidden_layers * 2
        assert len(dropout_nn.layers[1].layers) == 2

        for i, layer in enumerate(dropout_nn.layers[0].layers):
            if i % 2 == 0:
                isinstance(layer, tf.keras.layers.Dropout)
                layer.rate == rate[int(i / 2)]
            elif i % 2 == 1:
                isinstance(layer, tf.keras.layers.Dense)
                assert layer.units == units
                assert layer.activation == activation or layer.activation.__name__ == activation

        assert isinstance(dropout_nn.layers[1].layers[0], tf.keras.layers.Dropout)
        assert dropout_nn.layers[1].layers[0].rate == rate[-1]

        assert isinstance(dropout_nn.layers[1].layers[-1], tf.keras.layers.Dense)
        assert dropout_nn.layers[1].layers[-1].units == int(np.prod(outputs.shape))
        assert dropout_nn.layers[1].layers[-1].activation == tf.keras.activations.linear


def test_dropout_network_can_be_compiled(
    dropout_network: DropoutNetwork, query_point_shape: List[int], observation_shape: List[int]
) -> None:
    """Checks that dropout networks are compilable."""

    example_data = empty_dataset(query_point_shape, observation_shape)
    inputs, outputs = get_tensor_spec_from_data(example_data)

    dropout_nn = dropout_network(inputs, outputs)

    dropout_nn.compile(tf.optimizers.Adam(), tf.losses.MeanSquaredError())

    assert dropout_nn.compiled_loss is not None
    assert dropout_nn.compiled_metrics is not None
    assert dropout_nn.optimizer is not None


def test_dropout(dropout_network: DropoutNetwork) -> None:
    """Tests the ability of architecture to dropout."""

    example_data = empty_dataset([1], [1])
    inputs, outputs = get_tensor_spec_from_data(example_data)

    dropout_nn = dropout_network(inputs, outputs, rate=0.999999999)
    dropout_nn.compile(tf.optimizers.Adam(), tf.losses.MeanSquaredError())

    outputs = [dropout_nn(tf.constant([[1.0]]), training=True) for _ in range(100)]
    npt.assert_almost_equal(
        0.0, np.mean(outputs), err_msg=f"{dropout_network} not dropping up to randomness"
    )

@pytest.mark.parametrize("rate", [1.5, -1.0])
def test_dropout_rate_raises_invalidargument_error(
    dropout_network: DropoutNetwork, rate: Any
) -> None:
    """Tests that value error is raised when given wrong probability rates"""
    with pytest.raises(InvalidArgumentError):
        example_data = empty_dataset([1], [1])
        inputs, outputs = get_tensor_spec_from_data(example_data)
        _ = dropout_network(inputs, outputs, rate=rate)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_dropout_network_dtype(dropout_network: DropoutNetwork, dtype: tf.DType) -> None:
    '''Tests that network can infer data type from the data'''
    x = tf.constant([[1]], dtype=tf.float16)
    inputs, outputs = tf.TensorSpec([1], dtype), tf.TensorSpec([1], dtype)
    dropout_nn = dropout_network(inputs, outputs)

    assert dropout_nn(x).dtype == dtype

def test_dropout_network_accepts_scalars(dropout_network: DropoutNetwork) -> None:
    '''Tests that network can handle scalar inputs with ndim = 1 instead of 2'''
    example_data = empty_dataset([1, 1], [1, 1])
    inputs, outputs = get_tensor_spec_from_data(example_data)
    dropout_nn = dropout_network(inputs, outputs)

    test_points = tf.linspace(-1, 1, 100)

    assert dropout_nn(test_points).shape == (100, 1)
