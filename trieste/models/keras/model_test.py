# %%
# %load_ext autoreload
# %autoreload 2
import tensorflow as tf
import trieste
tf.get_logger().setLevel("ERROR")

from trieste.data import Dataset
from trieste.models.keras import (
    DeepEvidentialNetwork, 
    DeepEvidentialRegression,
    build_vanilla_keras_deep_evidential, 
    get_tensor_spec_from_data
)



# %%

x = tf.expand_dims(tf.linspace(0, 10, 100), axis=-1)
y = - x ** 2 + 4
data = Dataset(x, y)

query_point = tf.constant([[2.], [3.]])

new_data = tf.expand_dims(tf.linspace(3, 5, 10), axis=-1)

# %%
evidential_network = build_vanilla_keras_deep_evidential(data)
deep_evidential = DeepEvidentialRegression(evidential_network)

deep_evidential.optimize(data)
deep_evidential.sample(query_point, 4)
