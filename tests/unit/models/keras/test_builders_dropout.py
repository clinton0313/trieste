from typing import Sequence, Union

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import empty_dataset
from trieste.models.keras.architectures import DropConnectNetwork, DropoutNetwork
from trieste.models.keras.builders import build_vanilla_keras_mcdropout
from trieste.models.keras.layers import DropConnect


@pytest.mark.parametrize("units, activation", [(10, "relu"), (50, tf.keras.activations.tanh)])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("rate", [0.1, 0.9])
@pytest.mark.parametrize("dropout", [DropoutNetwork, DropConnectNetwork])
def test_build_vanilla_keras_mcdropout(
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    rate: float,
    dropout: str,
) -> None:
    example_data = empty_dataset([1], [1])
    mcdropout = build_vanilla_keras_mcdropout(
        example_data, num_hidden_layers, units, activation, rate, dropout
    )

    assert mcdropout.built
    assert isinstance(mcdropout, dropout)
    assert len(mcdropout.layers) == 2

    # Check Hidden Layers
    if num_hidden_layers > 0:
        for i, layer in enumerate(mcdropout.layers[0].layers):
            if dropout == DropConnectNetwork:
                assert len(mcdropout.layers[0].layers) == num_hidden_layers
                assert isinstance(layer, DropConnect)
                assert layer.units == units
                assert layer.activation == activation or layer.activation.__name__ == activation
            elif dropout == DropoutNetwork:
                assert len(mcdropout.layers[0].layers) == num_hidden_layers * 2
                if i % 2 == 0:
                    assert isinstance(layer, tf.keras.layers.Dropout)
                    assert layer.rate == rate
                elif i % 2 == 1:
                    assert isinstance(layer, tf.keras.layers.Dense)
                    assert layer.units == units
                    assert layer.activation == activation or layer.activation.__name__ == activation

    # Check Output Layers
    if dropout == DropConnectNetwork:
        assert isinstance(mcdropout.layers[1], DropConnect)
        assert mcdropout.layers[1].rate == rate
    elif dropout == DropoutNetwork:
        assert isinstance(mcdropout.layers[1].layers[0], tf.keras.layers.Dropout)
        assert mcdropout.layers[1].layers[0].rate == rate
        assert isinstance(mcdropout.layers[1].layers[1], tf.keras.layers.Dense)
