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
np.random.seed(1795)
tf.random.set_seed(1795)
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

# model.optimize(data)

# # %%
# import matplotlib.pyplot as plt


# # test data that includes extrapolation points
# test_points = tf.linspace(-6, 6, 1000)

# # generating a plot with ground truth function, mean prediction and 3 standard
# # deviations around it
# plt.scatter(inputs, outputs, marker=".", alpha=0.6, color="red", label="data")
# plt.plot(
#     test_points, objective(test_points, False), color="blue", label="function"
# )
# y_hat, y_var = model.predict(test_points)
# y_hat_minus_3sd = y_hat - 3 * tf.math.sqrt(y_var)
# y_hat_plus_3sd = y_hat + 3 * tf.math.sqrt(y_var)
# plt.plot(test_points, y_hat, color="gray", label="model $\mu$")
# plt.fill_between(
#     test_points,
#     tf.squeeze(y_hat_minus_3sd),
#     tf.squeeze(y_hat_plus_3sd),
#     color="gray",
#     alpha=0.5,
#     label="$\mu -/+ 3SD$",
# )
# plt.ylim([-100, 100])
# plt.show()

# %%

# %%
from trieste.objectives import (
    michalewicz_2,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_2_SEARCH_SPACE,
)
from util.plotting_plotly import plot_function_plotly

search_space = MICHALEWICZ_2_SEARCH_SPACE
function = michalewicz_2
MINIMUM = MICHALEWICZ_2_MINIMUM
MINIMIZER = MICHALEWICZ_2_MINIMUM

# we illustrate the 2-dimensional Michalewicz function
fig = plot_function_plotly(
    function, search_space.lower, search_space.upper, grid_density=100
)
fig.update_layout(height=800, width=800)
fig.show()

# %%

from trieste.objectives.utils import mk_observer

num_initial_points = 20

initial_query_points = search_space.sample(num_initial_points)
observer = trieste.objectives.utils.mk_observer(function)
initial_data = observer(initial_query_points)
# %%
def build_model(data: Dataset) -> DirectEpistemicUncertaintyPredictor:

    ensemble_params = {
        "ensemble_size": 5,
        "num_hidden_layers": 4,
        "units": 25,
        "activation": "relu",
        "independent_normal": False
    }

    uncertainty_params = {
        "num_hidden_layers": 4,
        "units": 256,
        "activation": "relu"
    }

    f_keras_ensemble, e_predictor = build_vanilla_deup(
        data, f_model_args=ensemble_params, e_model_args=uncertainty_params
    )

    fit_args = {
        "batch_size": 10,
        "epochs": 1000,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100)
        ],
        "verbose": 0,
    }
    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.001), fit_args)

    f_ensemble = DeepEnsemble(f_keras_ensemble, optimizer)

    deup = DirectEpistemicUncertaintyPredictor(
        model={"f_model": f_ensemble, "e_model": e_predictor},
        optimizer=optimizer, init_buffer=True
    )

    return deup

# %%
model = build_model(initial_data)
# %%

from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

# grid_size = 2000
# num_samples = 4

# # note that `DiscreteThompsonSampling` by default uses `ExactThompsonSampler`
# acquisition_rule = DiscreteThompsonSampling(grid_size, num_samples)

acquisition_rule = EfficientGlobalOptimization(num_query_points=1)
# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 3

# The Keras interface does not currently support using `track_state=True` which saves the model
# in each iteration. This will be addressed in a future update.
result = bo.optimize(
    num_steps,
    initial_data,
    model,
    acquisition_rule=acquisition_rule,
    track_state=False,
)
dataset = result.try_get_final_dataset()

# %%
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"Minimizer query point: {query_points[arg_min_idx, :]}")
print(f"Minimum observation: {observations[arg_min_idx, :]}")
print(f"True minimum: {MINIMUM}")

# %%
from util.plotting_plotly import add_bo_points_plotly

fig = plot_function_plotly(
    function,
    search_space.lower,
    search_space.upper,
    grid_density=100,
    alpha=0.7,
)
fig.update_layout(height=800, width=800)

fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
    fig=fig,
)
fig.show()

# %%
