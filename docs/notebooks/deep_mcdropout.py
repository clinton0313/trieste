
# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import trieste

# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
tf.get_logger().setLevel("ERROR")



# %%
np.random.seed(1794)
tf.random.set_seed(1794)
tf.keras.backend.set_floatx("float64")


# %% [markdown]
# ## MC dropout

# %%
from trieste.space import Box
from trieste.data import Dataset


def objective(x, error=True):
    y = tf.pow(x, 3)
    if error:
        y += tf.random.normal(x.shape, 0, 3, dtype=x.dtype)
    return y


num_points = 20

# we define the [-4,4] interval using a `Box` search space that has convenient sampling methods
search_space = Box([-4], [4])
inputs = search_space.sample_sobol(num_points)
outputs = objective(inputs)
data = Dataset(inputs, outputs)


# %% [markdown]
# Next we define a MC dropout model and train it.

# %%
from trieste.models.keras import (
    DeepDropout,
    KerasPredictor,
    build_vanilla_keras_mcdropout,
)
from trieste.models.optimizer import KerasOptimizer


def build_cubic_model(data: Dataset) -> DeepDropout:
    num_hidden_layers = 3
    num_nodes = 100
    activation = "relu"
    rate = 0.2

    dropout_nn = build_vanilla_keras_mcdropout(data)

    fit_args = {
        "batch_size": 10,
        "epochs": 1000,
        "verbose": 0,
    }
    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.01), fit_args)

    return DeepDropout(dropout_nn, optimizer)


# building and optimizing the model
model = build_cubic_model(data)
model.optimize(data)


# %% [markdown]
# Let's illustrate the results
# %%
import matplotlib.pyplot as plt


# test data that includes extrapolation points
test_points = tf.linspace(-6, 6, 1000)

# generating a plot with ground truth function, mean prediction and 3 standard
# deviations around it
plt.scatter(inputs, outputs, marker=".", alpha=0.6, color="red", label="data")
plt.plot(
    test_points, objective(test_points, False), color="blue", label="function"
)
y_hat, y_var = model.predict(test_points)
y_hat_minus_3sd = y_hat - 3 * tf.math.sqrt(y_var)
y_hat_plus_3sd = y_hat + 3 * tf.math.sqrt(y_var)
plt.plot(test_points, y_hat, color="gray", label="model $\mu$")
plt.fill_between(
    test_points,
    tf.squeeze(y_hat_minus_3sd),
    tf.squeeze(y_hat_plus_3sd),
    color="gray",
    alpha=0.5,
    label="$\mu -/+ 3SD$",
)
plt.ylim([-100, 100])
plt.show()

# %%
