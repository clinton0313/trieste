import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.python.eager import context
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import (
    embedding_ops,
    gen_math_ops,
    math_ops,
    nn_ops,
    sparse_ops,
    standard_ops,
)


class DropConnect(Dense):
    """
    This layer creates is a fully connected Dense layer that employs dropout to the
    weights of each unit rather than the unit inputs themselves as is done in standard
    Dropout. This layer is meant to be used as layers in
    :class:`~trieste.models.keras.architectures.DropConnectNetwork` architecture,
    """

    def __init__(self, rate: float = 0.3, *args, **kwargs):
        """
        :param units: Number of units to use in the layer.
        :param rate: The probability of dropout applied to each weight of a Dense Keras layer.
        :param *args: Args passed to Dense Keras class.
        "param **kwargs: Keyword arguments passed to Dense Keras class
        :raise ValueError: ``rate`` is not a valid probability
        """
        self.rate = rate
        super().__init__(*args, **kwargs)

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, rate: float):
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Rate needs to be a valid probability, instead got {rate}")
        else:
            self._rate = rate

    def call(self, inputs, training=False):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        # Drop Connect Code to mask the kernel
        if training:
            kernel = tf.nn.dropout(self.kernel, self.rate)
        else:
            kernel = self.kernel

        # Code and comments below from Tensorflow Dense Class
        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul operation for
            # large sparse input tensors. The op will result in a sparse gradient, as
            # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, sparse_tensor.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = sparse_tensor.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape,
                )
                weights = inputs
                outputs = embedding_ops.embedding_lookup_sparse_v2(
                    kernel, ids, weights, combiner="sum"
                )
            else:
                outputs = gen_math_ops.MatMul(a=inputs, b=kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
        # Reshape the output back to the original ndim of the input.
        if not context.executing_eagerly():
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [kernel.shape[-1]]
            outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = nn_ops.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
