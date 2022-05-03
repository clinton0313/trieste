# %% [markdown]
# # Bayesian optimization with deep evidential regression
#
# Gaussian processes as a surrogate models are hard to beat on smaller datasets and optimization budgets. However, they scale poorly with amount of data, cannot easily capture non-stationarities and they are rather slow at prediction time. Here we show how uncertainty-aware neural networks can be effective alternative to Gaussian processes in Bayesian optimisation, in particular for large budgets, non-stationary objective functions or when predictions need to be made quickly.
#
# Check out our tutorial on [Deep Gaussian Processes for Bayesian optimization](deep_gaussian_processes.ipynb) as another alternative model type supported by Trieste that can model non-stationary functions (but also deal well with small datasets).
#
# Let's start by importing some essential packages and modules.

import numpy as np
import os
import tensorflow as tf
import trieste
from trieste.acquisition.rule import EfficientGlobalOptimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from trieste.data import Dataset
from trieste.models.keras import (
    DeepEvidentialRegression,
    build_vanilla_keras_deep_evidential, 
    get_tensor_spec_from_data
)

from trieste.models.optimizer import KerasOptimizer

# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
tf.get_logger().setLevel("ERROR")

# %% [markdown]
# Trieste works with `tf.float64` as a default. It is advised to set the Keras backend float to the same value using `tf.keras.backend.set_floatx()`. Otherwise code might crash with a ValueError!

# %%
# np.random.seed(1794)
# tf.random.set_seed(1794)
tf.keras.backend.set_floatx("float64")

# %% [markdown]
# ## Deep Evidential Regression
#
# Deep neural networks typically output only mean predictions, not posterior distributions as probabilistic models such as Gaussian processes do. Posterior distributions encode mean predictions, but also *epistemic* uncertainty - type of uncertainty that stems from model misspecification, and which can be eliminated with further data. Aleatoric uncertainty that stems from stochasticity of the data generating process is not contained in the posterior, but can be learned from the data. Bayesian optimization requires probabilistic models because epistemic uncertainty plays a key role in balancing between exploration and exploitation.
#
# Recently, however, there has been some development of uncertainty-aware deep neural networks. Deep evidential regression is a deterministic flavour of these types of networks <(cite data-cite="amini2020evidential"/>). Main ingredients are probabilistic feed-forward network, where the final layer outputs evidential parameters, training with maximum likelihood,  and a custom regularizer instead of typical root mean square error. 
#
# Monte carlo dropout (<cite data-cite="gal2016dropout"/>), Bayes-by-backprop (<cite data-cite="blundell2015weight"/>) or deep ensembles (<cite data-cite="lakshminarayanan2016simple"/>) are some of the other types of uncertainty-aware deep neural networks. Deep evidential regression has the advantage of being deterministic which makes its training and prediction very fast. Furthermore, it is able to distinguish between aleatoric and epsitemic uncertainty analyitically.
#
# Check out our other tutorials for [Monte Carlo Dropout with Bayesian Optimization](deep_mcdropout.pct.py) and [Deep Ensembles for Bayesian Optimization](deep_ensembles.pct.py) as other alternative model types supported by Trieste that can model non-stationary functions.

# %% [markdown]
# ### How good is uncertainty representation of deep evidential regression?
#
# We will use a simple one-dimensional toy problem introduced by <cite data-cite="hernandez2015probabilistic"/>, which was used in <cite data-cite="amini2020evidential"/> to provide some illustrative evidence that deep envidential regression can do a good job of estimating uncertainty. We will replicate this exercise here.
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


num_points = 100

# we define the [-4,4] interval using a `Box` search space that has convenient sampling methods
search_space = Box([-4], [4])
inputs = search_space.sample_sobol(num_points)
outputs = objective(inputs)
data = Dataset(inputs, outputs)


# %% [markdown]
# Next we define a deep evidential regression model and train it. Trieste supports neural network models defined as TensorFlow's Keras models. Since creating deep evidential models in Keras can be somewhat involved, Trieste provides some basic architectures. Here we use the `build_vanilla_keras_deep_evidential` function which builds a simple deep evidential neural network in Keras. It uses sensible defaults for many parameters and finally returns a model of `DeepEvidentialNetwork` class.
#
# As with other supported types of models (e.g. Gaussian process models from GPflow), we cannot use `DeepEvidentialNetwork` directly in Bayesian optimization routines, we need to pass it through an appropriate wrapper, `DeepEvidentialRegression` wrapper in this case. One difference with respect to other model types is that we need to use a Keras specific optimizer wrapper `KerasOptimizer` where we need to specify a stochastic optimizer (Adam is used by default, but we can use other stochastic optimizers from TensorFlow), a custom loss function appropriate to deep evidential regression is automatically initialized,  and we can provide custom arguments for the Keras `fit` method (here we modify the default arguments; check [Keras API documentation](https://keras.io/api/models/model_training_apis/#fit-method) for a list of possible arguments).
#
# For the cubic function toy problem we use a relatively small architecture of 2 layers and 50 units. We use a ``reg_weight`` of 1e-2 to regularize the model confidence without any iterative updating of this weight. This works well in this specific intialization, but the quantification of uncertainty in this model is highly regularized by this parameter (or equivalently the ``maxi_rate`` parameter that controls the iterative search for ``reg_weight``). Too high of a weight will lead to over-regularization of model confidence and exploding uncertainty whereas too small of a weight can lead to overconfidence. 


# %%
from trieste.models.keras import (
    DeepEvidentialRegression,
    build_vanilla_keras_deep_evidential,
)
from trieste.models.optimizer import KerasOptimizer


def build_cubic_model(data: Dataset) -> DeepEvidentialRegression:
    num_hidden_layers = 2
    num_nodes = 50

    keras_ensemble = build_vanilla_keras_deep_evidential(
        data, num_hidden_layers, num_nodes
    )

    fit_args = {
        "batch_size": 10,
        "epochs": 1000,
        "verbose": 0,
        "callbacks": tf.keras.callbacks.EarlyStopping()
    }
    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.01), fit_args)

    return DeepEvidentialRegression(keras_ensemble, optimizer, reg_weight = 1e-2, maxi_rate=0.)


# building and optimizing the model
model = build_cubic_model(data)
model.optimize(data)

# %% [markdown]
# Let's illustrate the results of the model training. We create a test set that includes points outside the interval on which the model has been trained. These extrapolation points are a good test of model's representation of uncertainty. What would we expect to see? Bayesian inference provides a reference frame. Predictive uncertainty should increase the farther we are from the training data and the predictive mean should start returning to the prior mean (assuming standard zero mean function).
#
# We can see in the figure below that predictive distribution of deep evidential regression indeed can exhibit these features. The figure also replicates fairly well the results from <cite data-cite="amini2020evidential"/>. This gives us some assurance that deep evidential regression might provide uncertainty that is good enough for trading off between exploration and exploitation in Bayesian optimization. But, as previously noted, these results are highly contingent on the selection of an appropriate ``reg_weight``. Furthermore, we did not find that results with this model are consistent across different random initializations. Despite these disconcerting results, the model in practice works extremely well in Bayesian Optimization. 

#%%

import matplotlib.pyplot as plt

def plot_scatter_with_var(query_points, y_pred, y_var, ax, n_stds=3, max_alpha = 0.7):
    x = tf.squeeze(query_points)
    y = tf.squeeze(y_pred)
    std = tf.squeeze(y_var**0.5) 
    ax.plot(x, y, color="black", label="predictions")
    for k in range(1, n_stds + 1):
        upper_std = y + k * std
        lower_std = y - k * std
        ax.fill_between(x, upper_std, lower_std, alpha = max_alpha/k, color="tab:blue")

def plot_cubic(train_data:Dataset, test_data: Dataset, ood_predictions):

    fig, ax = plt.subplots(figsize=(14,10))

    ax.axvline(-4, color="grey", linestyle="dashed")
    ax.axvline(4, color="grey", linestyle="dashed")
    ax.plot(test_data.query_points, test_data.observations, color="red", linestyle="dashed")
    plot_scatter_with_var(test_data.query_points, ood_predictions[0], np.log(ood_predictions[1]), ax=ax)
    ax.scatter(train_data.query_points, train_data.observations, color="tab:red", s=20, alpha = 0.8)
    ax.set_ylim(-150, 150)
    return fig

x_test = tf.expand_dims(tf.linspace(-6, 6, 1000), axis=-1)
y_test = objective(x_test, error=False)
test_data = Dataset(x_test, y_test)
ood_predictions = model.predict(x_test, aleatoric=False)

fig = plot_cubic(data, test_data, ood_predictions)

#%% TESTING

evidential_output = model.model(x_test)[0]
gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=1)
sample_var = tf.math.reduce_variance(model.sample(x_test, 3000), axis=0)
aleatoric = beta/(alpha - 1)
epistemic = beta/((alpha - 1) * v)

total = aleatoric + epistemic

#%%

f, a = plt.subplots(figsize = (12, 12))
a.plot(x_test, sample_var, color="black", label="Sample Variance")
a.plot(x_test, np.log(epistemic + 1), color="red", label="Epistemic")
a.plot(x_test, np.log(aleatoric + 1), color="blue", label="Aleatoric")
a.plot(x_test, np.log(aleatoric * epistemic + 1), color = "orange", label="Total Uncertainty")
a.legend()

# %% [markdown]
# ## Non-stationary toy problem
#
# Now we turn to a somewhat more serious synthetic optimization problem. We want to find the minimum of the two-dimensional version of the [Michalewicz function](https://www.sfu.ca/~ssurjano/michal.html). Even though we stated that deep evidential regression should be used with larger budget sizes, here we will show them on a small dataset to provide a problem that is feasible for the scope of the tutorial.

# The Michalewicz function is defined on the search space of $[0, \pi]^2$. Below we plot the function over this space. The Michalewicz function is interesting case for deep evidential regression as it features sharp ridges that are difficult to capture with Gaussian processes. This occurs because lengthscale parameters in typical kernels cannot easily capture both ridges (requiring smaller lengthscales) and fairly flat areas everywhere else (requiring larger lengthscales).


# %%
from trieste.objectives import (
    michalewicz_2,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_2_SEARCH_SPACE,
    scaled_branin,
    SCALED_BRANIN_MINIMUM,
    BRANIN_SEARCH_SPACE
)
from util.plotting_plotly import plot_function_plotly

search_space = MICHALEWICZ_2_SEARCH_SPACE
function = michalewicz_2
MINIMUM = MICHALEWICZ_2_MINIMUM
MINIMIZER = MICHALEWICZ_2_MINIMUM


search_space = BRANIN_SEARCH_SPACE
function = scaled_branin
MINIMUM = SCALED_BRANIN_MINIMUM
MINIMIZER = SCALED_BRANIN_MINIMUM

# we illustrate the 2-dimensional Michalewicz function
# fig = plot_function_plotly(
#     function, search_space.lower, search_space.upper, grid_density=100
# )
# fig.update_layout(height=800, width=800)
# fig.show()


# %% [markdown]
# ## Initial design
#
# We set up the observer as usual, using Sobol sampling to sample the initial points.

# %%
from trieste.objectives.utils import mk_observer

num_initial_points = 1

initial_query_points = search_space.sample(num_initial_points)
observer = trieste.objectives.utils.mk_observer(function)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Modelling the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. Here we use a deep ensemble instead of a typical probabilistic model. Same as above we use the `build_vanilla_keras_deep_evidential` function to build a deep evidential neural network in Keras and wrap it with a `DeepEvidentialRegression` wrapper so it can be used in Trieste's Bayesian optimization loop.
#
# Some notes on choosing the model architecture are necessary. Unfortunately, choosing an architecture that works well for small datasets, a common setting in Bayesian optimization, is not easy. Here we do demonstrate it can work with smaller datasets, but choosing the architecture and model optimization parameters was a lengthy process that does not necessarily generalize to other problems. Hence, we advise to use deep evidential regression with larger datasets and ideally large batches so that the model is not retrained after adding a single point.
#
# We can offer some practical advices, however. Architecture parameters like number of nodes and number of layers will heavily affect the network's ability to learn the function. Too small of a network and it will be unable to accurately approximate the function, too large and the network will take too long to train. The model is also highly sensitive to the ``reg_weight`` parameter which balances the log-likelihood loss with the regularization term. In practice the use of ``maxi_rate`` to automatically search for a good ``reg_weight`` provides easier hyperparameter tuning. If we suspect the objective function is more complex these numbers should be increased slightly. With regards to model optimization we advise using a lot of epochs, typically at least 1000, and potentially higher learning rates. Ideally, every once in a while capacity should be increased to be able to use larger amount of data more effectively. Unfortunately, there is almost no research literature that would guide us in how to do this properly.
#
# Interesting alternative to a manual architecture search is to use a separate Bayesian optimization process to optimize the architecture and model optimizer parameters (see recent work by <cite data-cite="kadra2021well"/>). This optimization is much faster as it optimizes model performance. It would slow down the original optimization, so its worthwhile only if optimizing the objective function is much more costly.
#
# Below we change the `build_model` function to adapt the model slightly for the Michalewicz function. Since it's a more complex function we increase the number of hidden layers but keep the number of nodes per layer on the lower side. Note the large number of epochs

# %%

def build_model(data: Dataset, num_hidden_layers: int = 4, num_nodes: int = 200, **model_args) -> DeepEvidentialRegression:

    deep_evidential = build_vanilla_keras_deep_evidential(
        data, num_hidden_layers, num_nodes
    )

    fit_args = {
        "batch_size": 10,
        "epochs": 1000,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100, restore_best_weights=True)
        ],
        "verbose": 0,
    }
    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.001), fit_args)

    return DeepEvidentialRegression(
        deep_evidential, 
        optimizer, 
        **model_args
    )

def parse_rate(rate:float)-> str:
    out = str(rate)
    if out == "0":
        return out
    elif out.count("e") == 1:
        out = out.replace("-", "").replace("0", "")
        return out
    else:
        zeros = out.count("0")
        out = out.replace("0", "").replace(".", "")
        return out + "e" + str(zeros)


def hacky_sim(num_hidden_layers, num_nodes, reg_maxi: tuple, seed = 0):
    # building and optimizing the model
    np.random.seed(seed)
    tf.random.set_seed(seed) 
    reg_weight = reg_maxi[0]
    maxi_rate = reg_maxi[1]
    fig_title = f"TS(4) log_norm2 epistemic layers: {num_hidden_layers}, nodes: {num_nodes}, reg_maxi: {reg_maxi} seed: {seed}"
    save_title = f"TS(4)_log_norm2_l{num_hidden_layers}_n{num_nodes}_r{parse_rate(reg_weight)}_m{parse_rate(maxi_rate)}_branin{seed}"

    fig_path = f"/home/clinton/Documents/bse/masters_thesis/trieste/notebooks/der_figs/branin"
    os.makedirs(fig_path, exist_ok = True)

    initial_query_points = search_space.sample(num_initial_points)
    observer = trieste.objectives.utils.mk_observer(function)
    initial_data = observer(initial_query_points)
    model = build_model(initial_data, num_hidden_layers, num_nodes, reg_weight=reg_weight, maxi_rate=maxi_rate)


    # %% [markdown]
    # ## Run the optimization loop
    #
    # In Bayesian optimization we use an acquisition function to choose where in the search space to evaluate the objective function in each optimization step. Deep Evidential Regression model uses a feed forward network whose parameters act to approximate an evidential higher order distribution allowing it to output both aleatoric and epsitemic uncertainty. This means that many acquisition functions may be used such as Expected improvement (see `ExpectedImprovement`), Lower confidence bound (see `NegativeLowerConfidenceBound`) or Thompson sampling (see `ExactThompsonSampling`).
    #
    # Here we will illustrate Deep Evidential Regression with a Thompson sampling acquisition function. We use a discrete Thompson sampling strategy that samples a fixed number of points (`grid_size`) from the search space and takes a certain number of samples at each point based on the model posterior (`num_samples`, if more than 1 then this is a batch strategy).


    # %%
    from trieste.acquisition.rule import DiscreteThompsonSampling

    grid_size = 2000
    num_samples = 4

    # note that `DiscreteThompsonSampling` by default uses `ExactThompsonSampler`
    acquisition_rule = DiscreteThompsonSampling(grid_size, num_samples)
    # acquisition_rule = EfficientGlobalOptimization()

    
    # %% [markdown]
    # We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method.
    #
    # Note that the optimization might take a while!

    # %%
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

    num_steps = 25

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
    # We can visualise how the optimizer performed as a three-dimensional plot. Crosses mark the initial data points while dots mark the points chosen during the Bayesian optimization run. The points are colored from purple to yellow in order of their query from earliest to latest. You can see that there are some samples on the flat regions of the space, but the function very efficiently moves to the minimum where most of the points are concentrated.

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
    fig.update_layout(title = "fit " + fig_title)
    # fig.show()
    fig.write_html(os.path.join(fig_path, f"{save_title}_fit.html"))
    print(f"{save_title}_fit saved!")


    # %% [markdown]
    # We can visualise the model over the objective function by plotting the mean and 95% confidence intervals of its predictive distribution. Since it is not easy to choose the architecture of the deep ensemble we advise to always check with these types of plots whether the model seems to be doing a good job at modelling the objective function. In this case we can see that the model was able to capture the relevant parts of the objective function.

    # %%
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
    fig.update_layout(title="predict " + fig_title)
    # fig.show()
    fig.write_html(os.path.join(fig_path, f"{save_title}_predict.html"))
    print(f"{save_title}_predict saved!")

# # %% [markdown]
# # Finally, let's plot the regret over time, i.e. difference between the minimum of the objective function and lowest observations found by the Bayesian optimization over time. Below you can see two plots. The left hand plot shows the regret over time: the observations (crosses and dots), the current best (orange line), and the start of the optimization loop (blue line). The right hand plot is a two-dimensional search space that shows where in the search space initial points were located (crosses again) and where Bayesian optimization allocated samples (dots). The best point is shown in each (purple dot) and on the left plot you can see that we come very close to 0 which is the minimum of the objective function.

# # %%
# from util.plotting import plot_regret, plot_bo_points

# suboptimality = observations - MINIMUM.numpy()

# fig, ax = plt.subplots(1, 2)
# plot_regret(
#     suboptimality,
#     ax[0],
#     num_init=num_initial_points,
#     idx_best=arg_min_idx,
# )
# plot_bo_points(
#     query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx
# )
# ax[0].set_title("Minimum achieved")
# ax[0].set_ylabel("Regret")
# ax[0].set_xlabel("# evaluations")
# ax[1].set_ylabel("$x_2$")
# ax[1].set_xlabel("$x_1$")
# ax[1].set_title("Points in the search space")
# fig.show()

layers = [2]
num_nodes = [50]
reg_maxis = [
    # (1e-4, 0),
    # (1e-4, 0.01),
    (0, 0.001),
    (1e-4, 0)
]
seeds = range(10)

import itertools
for num_hidden_layers, num_nodes, reg_maxi, seed in itertools.product(layers, num_nodes, reg_maxis, seeds):
    hacky_sim(num_hidden_layers, num_nodes, reg_maxi, seed)

for seed in seeds:
    for num_hidden_layers, num_nodes, reg_maxi, seed in zip(
        [4, 2, 4, 2, 4, 2, 4, 4, 4, 4],
        [50, 100, 50, 50, 25, 100, 50, 25, 100, 50],
        zip(
            [0.01, 0.001, 0.001, 0, 0, 0, 0, 1e-5, 0.001, 0.01],
            [0, 0, 0, 0.001, 0.0001, 0.001, 0.001, 0.01, 0.01, 0.1]
        ),
        [seed for _ in range(10)]
    ):
        hacky_sim(num_hidden_layers, num_nodes, reg_maxi, seed)
# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)

# %%


#%%
#TESTING DURING DEVELOPMENT CODE

# import matplotlib
# import matplotlib.pyplot as plt
# import pickle
# import random

# def plot_scatter_with_var(query_points, y_pred, y_var, ax, n_stds=3, max_alpha = 0.7):
#     x = tf.squeeze(query_points)
#     y = tf.squeeze(y_pred)
#     std = tf.squeeze(y_var**0.5) #needs a square root missing...
#     ax.plot(x, y, color="black", label="predictions")
#     for k in range(1, n_stds + 1):
#         upper_std = y + k * std
#         lower_std = y - k * std
#         ax.fill_between(x, upper_std, lower_std, alpha = max_alpha/k, color="tab:blue")

# def cubic(x, noise=True):
#     y = tf.pow(x, 3)
#     if noise:
#         y += tf.random.normal(x.shape, 0, 3, dtype=x.dtype)
#     return y

# def gen_cubic_dataset(n, min, max, noise=True):
#     x = tf.linspace(min, max, n)
#     x = tf.cast(tf.expand_dims(tf.sort(x), axis=-1), dtype=tf.float64)
#     y = cubic(x, noise)
#     return Dataset(x,y)

# def gen_cubic_train_test(
#     n = 1000,
#     train_min = -4,
#     train_max = 4,
#     test_min = -7,
#     test_max = 7
# ):
#     train_data = gen_cubic_dataset(n, train_min, train_max, noise=True)
#     test_data = gen_cubic_dataset(n, test_min, test_max, noise=False)
#     return train_data, test_data

# def main_cubic(train_data, layers = 4, units=100, lr = 5e-3, fit_args = {}, **model_args):

#     evidential_network = build_vanilla_keras_deep_evidential(
#         train_data,
#         layers,
#         units
#     )

#     optimizer = KerasOptimizer(
#         tf.keras.optimizers.Adam(lr),
#         fit_args
#     )

#     deep_evidential = DeepEvidentialRegression(evidential_network, optimizer, **model_args)

#     return deep_evidential

# def plot_cubic(train_data:Dataset, test_data: Dataset, ood_predictions):

#     fig, ax = plt.subplots(figsize=(14,10))

#     ax.axvline(-4, color="grey", linestyle="dashed")
#     ax.axvline(4, color="grey", linestyle="dashed")
#     ax.plot(test_data.query_points, test_data.observations, color="red", linestyle="dashed")
#     plot_scatter_with_var(test_data.query_points, ood_predictions[0], ood_predictions[1], ax=ax)
#     ax.scatter(train_data.query_points, train_data.observations, color="tab:red", s=4, alpha = 0.6)
#     ax.set_ylim(-150, 150)
#     return fig

# #%%

# seed = 1234
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# with open("/home/clinton/Documents/bse/masters_thesis/trieste/notebooks/amini_cubic.pkl", "rb") as infile:
#     x_train, y_train, x_test, y_test = pickle.load(infile)

# x_train = tf.cast(x_train, dtype=tf.float64)
# x_test = tf.cast(x_test, dtype=tf.float64)
# y_train = tf.cast(y_train, dtype=tf.float64)
# y_test = tf.cast(y_test, dtype=tf.float64)


# train_data = Dataset(x_train, y_train)
# test_data = Dataset(x_test, y_test)

# # n=1000
# # train_data = gen_cubic_dataset(n, -4, 4, True)
# # test_data = gen_cubic_dataset(n, -7, 7, False)


# #%%

# fit_args = {
#                 "verbose": 0,
#                 "epochs": 5000,
#                 "batch_size": 128,
#                 "callbacks": [
#                     tf.keras.callbacks.EarlyStopping(
#                         monitor="loss", patience=200, restore_best_weights=True
#                     )
#                 ],
#             }

# # deep_evidential = main_cubic(
# #     train_data, 
# #     fit_args=fit_args,
# #     reg_weight=1e-2,
# #     maxi_rate=0.,
# #     verbose=1
# # )
# # #%%
# # deep_evidential.optimize(train_data)
# # predictions = deep_evidential.predict(train_data.query_points)
# # error = tf.abs(train_data.observations - predictions[0])
# # mean_error = tf.reduce_mean(error, axis=0)
# # print(f"mean abs error {mean_error}")
# # #%%
# # ood_predictions = deep_evidential.predict(test_data.query_points)
# # plot_cubic(train_data, test_data, ood_predictions)

# # # #%%
# # # #DIAGNOSIS
# # # gamma, lamb, alpha, beta = tf.split(deep_evidential.model(tf.expand_dims(test_data.query_points, axis=-1))[0], 4, axis=-1)

# # # names = ["gamma", "lambda", "alpha", "beta"]

# # # for name, output in zip(names, [gamma, lamb, alpha, beta]):
# # #     print(f"{name} has max: {np.max(output)} and min {np.min(output)}")

# # # %%
# # plt.plot(deep_evidential.model.history.history["loss"])
# # # plt.ylim(0, 100)
# # deep_evidential.model.history.history["loss"][-10:]

# # # %%

# # plt.plot(test_data.query_points,ood_predictions[1])
# # # %%
# # #Run the same cubic data over and over again to check the figs for
# # #consistency

# # %%

# def simulate_cubic(
#     prefix,
#     sims = 30, 
#     seed=1234, 
#     refresh_seed=False,
#     train_data = None,
#     test_data = None, 
#     refresh_data = False, 
#     n=1000,
#     **main_args
# ):
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     models = []
#     for i in range(sims):
#         if refresh_seed:
#             random.seed(i)
#             np.random.seed(i)
#             tf.random.set_seed(i)
#         if refresh_data:
#             train_data, test_data = gen_cubic_train_test(n)

#         deep_evidential = main_cubic(
#             train_data=train_data,
#             **main_args
#         )
#         deep_evidential.optimize(train_data)
#         ood_predictions = deep_evidential.predict(test_data.query_points)
#         fig = plot_cubic(train_data, test_data, ood_predictions)
    
#         models.append(deep_evidential)
#         fig.savefig(f"/home/clinton/Documents/bse/masters_thesis/trieste/notebooks/de_figs/{prefix}_fig{i}.png", facecolor="white", transparent=False)

#     return models

# models = simulate_cubic(
#     prefix="l2_reg_0_001",
#     train_data = train_data, 
#     test_data = test_data,
#     fit_args = fit_args,
#     reg_weight=1e-2,
#     maxi_rate=0.,
#     verbose=0
# )

#%%

