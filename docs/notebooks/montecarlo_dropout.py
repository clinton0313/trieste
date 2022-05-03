# %% [markdown]
# # Bayesian optimization with Monte Carlo Dropout
#
# Gaussian processes as surrogate models are hard to beat on smaller datasets and optimization budgets. However, they scale poorly with the amount of data, cannot easily capture non-stationarities and they are rather slow at prediction time. In these cases, the practitioner may use uncertainty-aware neural networks as an effective alternative to Gaussian processes in Bayesian optimisation, in particular for large budgets, non-stationary objective functions or when predictions need to be made quickly.
#
# This tutorial presents the implementation of Monte Carlo Dropout, a method to equip feedforward artificial neural networks with estimates of uncertainty. For an ensemble-based approach, see our tutorial [Bayesian optimization with deep ensembles](deep_gaussian_processes.ipynb). Also check out our tutorial on [Deep Gaussian Processes for Bayesian optimization](deep_gaussian_processes.ipynb) as another alternative model type supported by Trieste that can model non-stationary functions.
#
# Let's start by importing some essential packages and modules.
# %%

import os

from trieste.models.keras.architectures import DropConnectNetwork

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import trieste

# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
tf.get_logger().setLevel("ERROR")

# %% [markdown]
# Trieste works with `tf.float64` as a default. It is advised to set the Keras backend float to the same value using `tf.keras.backend.set_floatx()`. Otherwise code might crash with a ValueError!

np.random.seed(1794)
tf.random.set_seed(1794)
tf.keras.backend.set_floatx("float64")

# %% [markdown]
# ## Monte Carlo Dropout
#
# Deep neural networks are able to learn powerful representations of the data, mapping points from high-dimensional spaces to an array of outputs. However, this array is typicially a set of point predictions, and not the posterior distributions one can obtain from Guassian processes. The model is thus generally agnostic about the *epistemic* uncertainty embedded in the array of point predictions, the type of uncertainty that stems from model misspecification, and which can be learned and eliminated with further observations. This reducible source of uncertainty it critical to Bayesian optimization, as it plays a key role in balancing between exploration and exploitation of the data space.  
#
# In light of this limitation, a thriving literature has emerged in recent years that aims to develop theoretically-grounded uncertainty-aware neural networks. Monte Carlo Dropout, originally put forth by <cite data-cite="gal2016dropout"/>, is one such example. This approach repurposes the dropout regularization layer at testing time by computing Monte Carlo simulations of the same forward pass and using the variation generated due to the dropout layer to infer the model's uncertainty about a prediction. Good estimates of uncertainty makes neural networks with Monte Carlo dropout layers potentially attractive models for Bayesian optimization.

# %% [markdown]
# ### How good is uncertainty representation of neural networks with Monte Carlo dropout layers?
#
# We will use a simple one-dimensional toy problem introduced by <cite data-cite="hernandez2015probabilistic"/>, which was used in <cite data-cite="lakshminarayanan2016simple"/> to provide some illustrative evidence on the good performance of deep ensembles, which we use with Monte Carlo dropout to provide a comparison with the results in the [deep ensembles tutorial](deep_gaussian_processes.ipynb).
#
# The toy problem is a simple cubic function with some Normally distributed noise around it. We will randomly sample 20 input points from [-4,4] interval that we will use as a training data later on.

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
# Next we define a Monte Carlo dropout model and train it. Trieste supports neural network models defined as Tensorflow's Keras models. S

# %%
from trieste.models.keras import (
    MonteCarloDropout,
    KerasPredictor,
    build_vanilla_keras_mcdropout,
)
from trieste.models.optimizer import KerasOptimizer


def build_cubic_model(data: Dataset) -> MonteCarloDropout:
    num_hidden_layers = 3
    num_nodes = 300
    activation = "relu"
    rate = 0.2

    dropout_nn = build_vanilla_keras_mcdropout(
        data,
        num_hidden_layers=num_hidden_layers,
        units=num_nodes,
        activation=activation,
        rate=rate,
        dropout_network=DropConnectNetwork
    )

    fit_args = {
        "batch_size": 10,
        "epochs": 1000,
        "verbose": 0,
    }
    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.01), fit_args)

    return MonteCarloDropout(dropout_nn, optimizer, num_passes=100)


# building and optimizing the model
model = build_cubic_model(data)
model.optimize(data)

# %% [markdown]
# Let's illustrate the results of the model training. We create a test set that includes points outside the interval on which the model has been trained. These extrapolation points are a good test of model's representation of uncertainty. What would we expect to see? Bayesian inference provides a reference frame. Predictive uncertainty should increase the farther we are from the training data and the predictive mean should start returning to the prior mean (assuming standard zero mean function).
#
# We can see in the figure below that the predictive distribution of the neural network indeed exhibits these features. This gives us some assurance that neural networks with Monte Carlo dropout layers might provide uncertainty that is good enough for trading off between exploration and exploitation in Bayesian optimization.

# %%
import matplotlib.pyplot as plt

test_points = tf.linspace(-6, 6, 1000)

# generating a plot with ground truth function, mean prediction and 3 standard
# deviations around it
plt.scatter(inputs, outputs, marker=".", alpha=0.6, color="red", label="data")
plt.plot(test_points, objective(test_points, False), color="blue", label="function")
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


# %% [markdown]
# ## Non-stationary toy problem
#
# A more serious synthetic optimization problem is presented next. We want to find the minimum of the two-dimensional version of the [Michalewicz function](https://www.sfu.ca/~ssurjano/michal.html). Even though we stated that uncertainty-aware neural networks should be used with larger budget sizes, here we will show them on a small dataset to provide a problem that is feasible for the scope of the tutorial.
#
# The Michalewicz function is defined on the search space of $[0, \pi]^2$. Below we plot the function over this space. The Michalewicz function is an interesting case for our Monte Carlo dropout appraoch as it features sharp ridges that are difficult to capture with Gaussian processes. This occurs because lengthscale parameters in typical kernels cannot easily capture both ridges (requiring smaller lengthscales) and fairly flat areas everywhere else (requiring larger lengthscales).

from trieste.objectives import (
    michalewicz_2,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_2_SEARCH_SPACE,
)
from util.plotting_plotly import plot_function_plotly

function = michalewicz_2
search_space = MICHALEWICZ_2_SEARCH_SPACE
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
# We set up the observer as before, using Sobol sampling to sample the initial points.

# %%
from trieste.objectives.utils import mk_observer

num_initial_points = 20

initial_query_points = search_space.sample(num_initial_points)
observer = trieste.objectives.utils.mk_observer(function)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Modelling the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. Here we use forward passes of the network with Monte Carlo dropout layers instead of a typical probabilistic model. Same as above we use the `build_vanilla_keras_mcdropout` function to build the underlying network in Keras and wrap it with a `MonteCarloDropout` wrapper so it can be used in Trieste's Bayesian optimization loop.
#
# Unfortunately, choosing an architecture that works well for small datasets, a common setting in Bayesian optimization, is not easy. Here we do demonstrate it can work with smaller datasets, but choosing the model parameters required extensive testing and the values presented here may not necessarily generalize to other problems. As with deep ensembles, we advise to use Monte Carlo dropout with larger datasets.
#
# In particular, architecture parameters like the number of hidden layers, the amount of forward passes in prediction and the dropout rate affect the capacity of the model. Unlike deep ensembles, neural networks with Monte Carlo dropout layers appear to benefit from larger models and more densely populated layers. Regarding the dropout rate, a lower rate will produce more accurate point predictions, but the degree of uncertainty will also decrease, potentially difficulting the ability to explore during the Bayesian optimization routine. Unfortunately, there is almost no research literature that addresses these finetuning limitations.
#
# Interesting alternative to a manual architecture search is to use a separate Bayesian optimization process to optimize the architecture and model optimizer parameters (see recent work by <cite data-cite="kadra2021well"/>). This optimization is much faster as it optimizes model performance. It would slow down the original optimization, so its worthwhile only if optimizing the objective function is much more costly.
#
# Below we change the `build_model` function to adapt the model slightly for the Michalewicz function. Since it's a more complex function we increase the number of hidden layers but keep the number of nodes per layer on the lower side. Note the large number of epochs

# %%
def build_model(data: Dataset) -> MonteCarloDropout:
    num_hidden_layers = 5
    num_nodes = 300

    dropout_network = build_vanilla_keras_mcdropout(
        data, num_hidden_layers, num_nodes, rate=0.3, dropout_network=DropConnectNetwork
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

    return MonteCarloDropout(dropout_network, optimizer, num_passes=100)


# building and optimizing the model
model = build_model(initial_data)


# %% [markdown]
# ## Run the optimization loop
#
# In Bayesian optimization we use an acquisition function to choose where in the search space to evaluate the objective function in each optimization step. Our uncertainty-aware neural network sues forward passes with Monte Carlo dropout layers to generate estimates of model mean and variance at each evaluated point, which can then be retooled to work as a predictive posterior distribution. This means that any acquisition function can be used that requires only predictive mean and variance. For example, predictive mean and variance is sufficient for standard acquisition functions such as Expected improvement (see `ExpectedImprovement`), Lower confidence bound (see `NegativeLowerConfidenceBound`) or Thompson sampling (see `ExactThompsonSampling`). Some acquisition functions have additional requirements and these cannot be used (e.g. covariance between sets of query points, as in an entropy-based acquisition function `GIBBON`).
#
# Here we will illustrate Monte Carlo dropout with a Thompson sampling acquisition function. We use a discrete Thompson sampling strategy that samples a fixed number of points (`grid_size`) from the search space and takes a certain number of samples at each point based on the model posterior (`num_samples`, if more than 1 then this is a batch strategy).

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
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 30

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
# We can visualise how the optimizer performed as a three-dimensional plot. Crosses mark the initial data points while dots mark the points chosen during the Bayesian optimization run. Darker dots represent early model queries, whereas bright colored dots represent points chosen at a late stage in the run. You can see that there are some samples on the flat regions of the space, while most of the points are exploring the ridges, in particular in the vicinity of the minimum point.

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
# We can visualise the model over the objective function by plotting the mean and 95% confidence intervals of its predictive distribution. Since it is not easy to choose the architecture of the Monte Carlo dropout neural network we advise to always check with these types of plots whether the model seems to be doing a good job at modelling the objective function. In this case we can see that the model was able to capture the relevant parts of the objective function.

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
# ## Additional and alternative 2-D functions
# I have been testing with some additional functions, and will leave here some of the ones that seem to work best. TODO: Wrap these in functions to make them be a couple of lines total.
# %%
# 
from trieste.objectives import (
    dropwave,
    DROPWAVE_MINIMUM,
    DROPWAVE_SEARCH_SPACE,
    DROPWAVE_MINIMIZER,
    eggholder,
    EGGHOLDER_MINIMUM,
    EGGHOLDER_SEARCH_SPACE,
    EGGHOLDER_MINIMIZER
)
from util.plotting_plotly import plot_function_plotly

function = eggholder
search_space = EGGHOLDER_SEARCH_SPACE
MINIMUM = EGGHOLDER_MINIMUM
MINIMIZER = EGGHOLDER_MINIMIZER

# we illustrate the dropwave function
fig = plot_function_plotly(
    function, search_space.lower, search_space.upper, grid_density=100
)
fig.update_layout(height=800, width=800)
fig.show()

# %%
num_initial_points = 20

initial_query_points = search_space.sample(num_initial_points)
observer = trieste.objectives.utils.mk_observer(function)
initial_data = observer(initial_query_points)


# %%
def build_model(data: Dataset) -> MonteCarloDropout:
    num_hidden_layers = 5
    num_nodes = 300

    dropout_network = build_vanilla_keras_mcdropout(
        data, num_hidden_layers, num_nodes, rate=0.3, dropout_network=DropConnectNetwork
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

    return MonteCarloDropout(dropout_network, optimizer, num_passes=100)


model = build_model(initial_data)

from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

grid_size = 2000
num_samples = 4

acquisition_rule = DiscreteThompsonSampling(grid_size, num_samples)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 30

result = bo.optimize(
    num_steps,
    initial_data,
    model,
    acquisition_rule=acquisition_rule,
    track_state=False,
)
dataset = result.try_get_final_dataset()

query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"Minimizer query point: {query_points[arg_min_idx, :]}")
print(f"Minimum observation: {observations[arg_min_idx, :]}")
print(f"True minimum: {MINIMUM}")


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
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
