import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
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
    def __init__(self, rate: float = 0.5, *args, **kwargs):
        """
        :param units: Number of units to use in the layer.
        :param rate: The probability of dropout applied to each weight of a Dense Keras layer.
        :param *args: Args passed to Dense Keras class.
        "param **kwargs: Keyword arguments passed to Dense Keras class
        """
        self.rate = rate
        super().__init__(*args, **kwargs)

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, rate):
        if not 0. <= rate < 1.:
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

        # Code below from Tensorflow Dense Class
        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            if isinstance(inputs, sparse_tensor.SparseTensor):
                inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
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
