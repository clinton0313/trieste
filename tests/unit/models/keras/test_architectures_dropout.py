#%%
from typing import Any, List, Tuple, Union

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import empty_dataset
from trieste.models.keras import (
    DropConnectNetwork,
    get_tensor_spec_from_data,
    negative_log_likelihood,
)
from trieste.models.keras.architectures import MCDropoutNetwork
from trieste.models.keras.layers import DropConnect

@pytest.fixture(name="dropout_network", params=[MCDropoutNetwork])
def _dropout_network_fixture(request: Any) -> MCDropoutNetwork:
    return request.param

@pytest.fixture(name="query_point_shape", params = [[1], [5]])
def _query_point_shape_fixture(request: Any) -> List[int]:
    return request.param

@pytest.fixture(name="observation_shape", params = [[1], [2]])
def _observation_shape_fixture(request: Any) -> List[int]:
    return request.param

@pytest.mark.parametrize(
    "num_hidden_layers, rate", 
    [
        (1, 0.3),
        (3, 0.7),
        (5, [0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    ]
)
@pytest.mark.parametrize("units", [10, 50])
@pytest.mark.parametrize("activation", ["relu", tf.keras.activations.tanh])
def test_dropout_network_build_seems_correct(
    dropout_network: MCDropoutNetwork, 
    query_point_shape: List[int], 
    observation_shape: List[int], 
    num_hidden_layers: int,
    units: int,
    activation : Union[str, tf.keras.layers.Activation],
    rate: Union[float, int, List[Union[int, float]]]
) -> None:
    '''Tests the correct consturction of dropout network architectures'''
    
    example_data = empty_dataset(query_point_shape, observation_shape)
    inputs, outputs = get_tensor_spec_from_data(example_data)
    hidden_layer_args = [{"units": units, "activation": activation} for _ in range(num_hidden_layers)]
    
    dropout_nn = dropout_network(
        outputs,
        hidden_layer_args,
        rate
    )

    if not isinstance(rate, list):
        rate = [rate for _ in range(num_hidden_layers + 1)]

    # basics
    assert isinstance(dropout_nn, tf.keras.Model)

    # # check input and output shapes
    # assert dropout_nn.input_shape[1:] == tf.TensorShape(query_point_shape)
    # assert dropout_nn.output_shape[1:] == tf.TensorShape(observation_shape)

    # check the model has not been compiled
    assert dropout_nn.compiled_loss is None
    assert dropout_nn.compiled_metrics is None
    assert dropout_nn.optimizer is None

    # # check input layer
    # assert isinstance(dropout_nn.layers[0], tf.keras.layers.InputLayer)
    
    # # check correct number of layers and proerply constructed
    # if isinstance(dropout_nn, DropConnectNetwork):
    #     assert len(dropout_nn.layers) == 2 + num_hidden_layers
        
    #     for layer in dropout_nn.layers[1:-1]:
    #         assert isinstance(layer, DropConnect)
    #         assert layer.units == units
    #         assert layer.activation == activation or layer.activation.__name__ == activation
        
    #     assert isinstance(dropout_nn.layers[-1], DropConnect)
    
    # elif isinstance(dropout_nn, MCDropoutNetwork):
    #     assert len(dropout_nn.layers) == 1 + 2 * (num_hidden_layers + 1)
        
    #     for i, layer in enumerate(dropout_nn.layers[1:-1]):
    #         if i % 2 == 0:
    #             isinstance(layer, tf.keras.layers.Dropout)
    #             layer.rate == rate[int(i/2)]
    #         elif i % 2 == 1:
    #             isinstance(layer, tf.keras.layers.Dense)
    #             assert layer.units == units
    #             assert layer.activation == activation or layer.activation.__name__ == activation
        
    #     assert isinstance(dropout_nn.layers[-1], tf.keras.layers.Dense)
    
    # # check output layer activation
    # assert dropout_nn.layers[-1].activation == tf.keras.activations.linear

def test_dropout_network_can_be_compiled(
    dropout_network: MCDropoutNetwork, 
    query_point_shape: List[int], 
    observation_shape: List[int]
 ) -> None:
    '''Checks that dropout networks are compilable.'''

    example_data = empty_dataset(query_point_shape, observation_shape)
    inputs, outputs = get_tensor_spec_from_data(example_data)

    dropout_nn = dropout_network(outputs)

    dropout_nn.compile(tf.optimizers.Adam(), tf.losses.MeanSquaredError())

    assert dropout_nn.compiled_loss is not None
    assert dropout_nn.compiled_metrics is not None
    assert dropout_nn.optimizer is not None


def test_dropout(dropout_network: MCDropoutNetwork) -> None:
    '''Tests the ability of architecture to dropout.'''

    example_data = empty_dataset([1], [1])
    inputs, outputs = get_tensor_spec_from_data(example_data)

    dropout_nn = dropout_network(outputs, rate=0.999999999)
    dropout_nn.compile(tf.optimizers.Adam(), tf.losses.MeanSquaredError())

    outputs = [dropout_nn(tf.constant([[1.]]), training=True) for _ in range(100)]
    npt.assert_almost_equal(0., np.mean(outputs), err_msg=f"{dropout_network} not dropping up to randomness")


# %%
