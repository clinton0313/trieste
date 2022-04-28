# %% [markdown]

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import trieste

# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
tf.get_logger().setLevel("ERROR")


# %% [markdown]

# %%
np.random.seed(1794)
tf.random.set_seed(1794)
tf.keras.backend.set_floatx("float64")


# %% [markdown]

# %% [markdown]


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

# %%
from trieste.models.keras import (
    DeepEnsemble,
    DirectEpistemicUncertaintyPredictor,
    EpistemicUncertaintyPredictor,
    KerasPredictor,
    build_vanilla_keras_ensemble,
    build_vanilla_deup
)
from trieste.models.optimizer import KerasOptimizer


def build_cubic_model(data: Dataset) -> DirectEpistemicUncertaintyPredictor:
    num_hidden_layers = 4
    num_nodes = 256

    deup_args = {
        "num_hidden_layers": num_hidden_layers,
        "units": num_nodes,
        "activation": "relu"
    }

    f_ensemble, e_predictor = build_vanilla_deup(
        data=data, e_model_args=deup_args
    )

    fit_args = {
        "batch_size": 10,
        "epochs": 1000,
        "verbose": 0,
    }

    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.01), fit_args)

    f_ensemble = DeepEnsemble(f_ensemble, optimizer)

    deup = DirectEpistemicUncertaintyPredictor(
        model={"f_model": f_ensemble, "e_model": e_predictor},
        optimizer=optimizer, init_buffer=True
    )

    return deup


# building and optimizing the model
model = build_cubic_model(data)

model.optimize(data)

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

"""
It looks like I'll have to somehow create an ensemble with the two models
"""