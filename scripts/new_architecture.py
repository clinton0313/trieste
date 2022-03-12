#%%
import tensorflow as tf

from trieste.models.keras.architectures import MCDropoutNetwork
from tests.util.misc import empty_dataset
from trieste.models.keras import get_tensor_spec_from_data
from trieste.models.keras.builders import build_vanilla_keras_mcdropout

#%%

example_data = empty_dataset([5], [1])
inputs, outputs = get_tensor_spec_from_data(example_data)

x = tf.constant([[1.]])

#%%
class Test(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(3)
    
    def call(self, x):
        return self.dense1(x)
mc = MCDropoutNetwork(outputs)
test = Test()

#%%
test.compile(tf.optimizers.Adam(), tf.losses.MeanAbsoluteError())
mc.compile(tf.optimizers.Adam(), tf.losses.MeanAbsoluteError())
# %%
mc.fit(x,x)

#%%