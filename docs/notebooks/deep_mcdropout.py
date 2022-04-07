# %%
%load_ext autoreload
%autoreload 2

import os

from trieste.models.keras.architectures import DropConnectNetwork

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import trieste

# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
tf.get_logger().setLevel("ERROR")

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
    MCDropout,
    KerasPredictor,
    build_vanilla_keras_mcdropout,
)
from trieste.models.optimizer import KerasOptimizer


# def build_cubic_model(data: Dataset) -> MCDropout:
#     num_hidden_layers = 3
#     num_nodes = 300
#     activation = "relu"
#     rate = 0.2

#     dropout_nn = build_vanilla_keras_mcdropout(
#         data,
#         num_hidden_layers=num_hidden_layers,
#         units=num_nodes,
#         activation=activation,
#         rate=rate,
#         dropout_network=DropConnectNetwork
#     )

#     fit_args = {
#         "batch_size": 10,
#         "epochs": 1000,
#         "verbose": 0,
#     }
#     optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.01), fit_args)

#     return MCDropout(dropout_nn, optimizer, num_passes=100)


# # building and optimizing the model
# model = build_cubic_model(data)
# model.optimize(data)

# # %% [markdown]
# # Let's illustrate the results
# # %%
# import matplotlib.pyplot as plt

# test_points = tf.linspace(-6, 6, 1000)

# # generating a plot with ground truth function, mean prediction and 3 standard
# # deviations around it
# plt.scatter(inputs, outputs, marker=".", alpha=0.6, color="red", label="data")
# plt.plot(test_points, objective(test_points, False), color="blue", label="function")
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


# %% [markdown]
# ## Initial design
#
# We set up the observer as usual, using Sobol sampling to sample the initial points.

# %%
from trieste.objectives.utils import mk_observer

num_initial_points = 20

initial_query_points = search_space.sample(num_initial_points)
observer = trieste.objectives.utils.mk_observer(function)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Modelling the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. Here we use a deep ensemble instead of a typical probabilistic model. Same as above we use the `build_vanilla_keras_ensemble` function to build a simple ensemble of neural networks in Keras and wrap it with a `DeepEnsemble` wrapper so it can be used in Trieste's Bayesian optimization loop.
#
# Some notes on choosing the model architecture are necessary. Unfortunately, choosing an architecture that works well for small datasets, a common setting in Bayesian optimization, is not easy. Here we do demonstrate it can work with smaller datasets, but choosing the architecture and model optimization parameters was a lengthy process that does not necessarily generalize to other problems. Hence, we advise to use deep ensembles with larger datasets and ideally large batches so that the model is not retrained after adding a single point.
#
# We can offer some practical advices, however. Architecture parameters like the ensemble size, the number of hidden layers, the number of nodes in the layers and so on affect the capacity of the model. If the model is too large for the amount of data, it will be difficult to train the model and result will be a poor model that cannot be used for optimizing the objective function. Hence, with small datasets like the one used here, we advise to always err on the smaller size, one or two hidden layers, and up to 25 nodes per layer. If we suspect the objective function is more complex these numbers should be increased slightly. With regards to model optimization we advise using a lot of epochs, typically at least 1000, and potentially higher learning rates. Ideally, every once in a while capacity should be increased to be able to use larger amount of data more effectively. Unfortunately, there is almost no research literature that would guide us in how to do this properly.
#
# Interesting alternative to a manual architecture search is to use a separate Bayesian optimization process to optimize the architecture and model optimizer parameters (see recent work by <cite data-cite="kadra2021well"/>). This optimization is much faster as it optimizes model performance. It would slow down the original optimization, so its worthwhile only if optimizing the objective function is much more costly.
#
# Below we change the `build_model` function to adapt the model slightly for the Michalewicz function. Since it's a more complex function we increase the number of hidden layers but keep the number of nodes per layer on the lower side. Note the large number of epochs

# %%
def build_model(data: Dataset) -> MCDropout:
    num_hidden_layers = 5
    num_nodes = 700

    dropout_network = build_vanilla_keras_mcdropout(
        data, num_hidden_layers, num_nodes, rate=0.001
    )

    fit_args = {
        "batch_size": 32,
        "epochs": 1000,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100), 
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.3, patience=20)
            ],
        "verbose": 0
    }

    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.001), fit_args)

    return MCDropout(dropout_network, optimizer, num_passes=100)


# building and optimizing the model
model = build_model(initial_data)


# %% [markdown]
# ## Run the optimization loop

# %%
from trieste.acquisition.rule import DiscreteThompsonSampling

grid_size = 2000
num_samples = 4

# note that `DiscreteThompsonSampling` by default uses `ExactThompsonSampler`
acquisition_rule = DiscreteThompsonSampling(grid_size, num_samples)


# %% [markdown]
# We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method.
#
# Note that the optimization might take a while!

# %%

from time import time
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 25

# The Keras interface does not currently support using `track_state=True` which saves the model
# in each iteration. This will be addressed in a future update.

start = time()
result = bo.optimize(
    num_steps,
    initial_data,
    model,
    acquisition_rule=acquisition_rule,
    track_state=False,
)
dataset = result.try_get_final_dataset()

print(f"This took {round(time() - start, 3)} seconds!")


# %% [markdown]
# ## Explore the results
#
# We can now get the best point found by the optimizer. Note this isn't necessarily the point that was last evaluated.

# %%
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"Minimizer query point: {query_points[arg_min_idx, :]}")
print(f"Minimum observation: {observations[arg_min_idx, :]}")
print(f"True minimum: {MINIMUM}")


# %% [markdown]
# We can visualise how the optimizer performed as a three-dimensional plot. Crosses mark the initial data points while dots mark the points chosen during the Bayesian optimization run. You can see that there are some samples on the flat regions of the space, while most of the points are exploring the ridges, in particular in the vicinity of the minimum point.

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


# %% [markdown]
# We can visualise the model over the objective function by plotting the mean and 95% confidence intervals of its predictive distribution. Since it is not easy to choose the architecture of the deep ensemble we advise to always check with these types of plots whether the model seems to be doing a good job at modelling the objective function. In this case we can see that the model was able to capture the relevant parts of the objective function.

# %%
import matplotlib.pyplot as plt
from util.plotting import plot_regret
from util.plotting_plotly import plot_model_predictions_plotly

fig = plot_model_predictions_plotly(
    result.try_get_final_model(),
    search_space.lower,
    search_space.upper,
)

fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
    fig=fig,
    figrow=1,
    figcol=1,
)
fig.update_layout(height=800, width=800)
fig.show()


# %% [markdown]
# Finally, let's plot the regret over time, i.e. difference between the minimum of the objective function and lowest observations found by the Bayesian optimization over time. Below you can see two plots. The left hand plot shows the regret over time: the observations (crosses and dots), the current best (orange line), and the start of the optimization loop (blue line). The right hand plot is a two-dimensional search space that shows where in the search space initial points were located (crosses again) and where Bayesian optimization allocated samples (dots). The best point is shown in each (purple dot) and on the left plot you can see that we come very close to 0 which is the minimum of the objective function.

# %%
from util.plotting import plot_regret, plot_bo_points

suboptimality = observations - MINIMUM.numpy()

fig, ax = plt.subplots(1, 2)
plot_regret(
    suboptimality,
    ax[0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
)
plot_bo_points(
    query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx
)
ax[0].set_title("Minimum achieved")
ax[0].set_ylabel("Regret")
ax[0].set_xlabel("# evaluations")
ax[1].set_ylabel("$x_2$")
ax[1].set_xlabel("$x_1$")
ax[1].set_title("Points in the search space")
fig.show()


# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
