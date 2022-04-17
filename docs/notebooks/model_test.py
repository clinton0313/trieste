# %%

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import tensorflow as tf

from trieste.data import Dataset
from trieste.models.keras import (
    DeepEvidentialRegression,
    DeepEnsemble,
    build_vanilla_keras_ensemble,
    build_vanilla_keras_deep_evidential, 
    get_tensor_spec_from_data
)

from trieste.models.keras.utils import (
    build_deep_evidential_regression_loss,
    normal_inverse_gamma_negative_log_likelihood,
    normal_inverse_gamma_regularizer
)
from trieste.models.optimizer import KerasOptimizer

matplotlib.use("TkAgg")
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")


#%%

def plot_scatter_with_var(query_points, y_pred, y_var, ax, n_stds=3, max_alpha = 0.7):
    x = tf.squeeze(query_points)
    y = tf.squeeze(y_pred)
    std = tf.squeeze(y_var**0.5) #needs a square root missing...
    ax.plot(x, y, color="black", label="predictions")
    for k in range(1, n_stds + 1):
        upper_std = y + k * std
        lower_std = y - k * std
        ax.fill_between(x, upper_std, lower_std, alpha = max_alpha/k, color="tab:blue")

def cubic(x, noise=True):
    y = tf.pow(x, 3)
    if noise:
        y += tf.random.normal(x.shape, 0, 3, dtype=x.dtype)
    return y

def gen_cubic_dataset(n, min, max, noise=True):
    x = tf.linspace(min, max, n)
    x = tf.cast(tf.expand_dims(tf.sort(x), axis=-1), dtype=tf.float64)
    y = cubic(x, noise)
    return Dataset(x,y)

def gen_cubic_train_test(
    n = 1000,
    train_min = -4,
    train_max = 4,
    test_min = -7,
    test_max = 7
):
    train_data = gen_cubic_dataset(n, train_min, train_max, noise=True)
    test_data = gen_cubic_dataset(n, test_min, test_max, noise=False)
    return train_data, test_data

def main_cubic(train_data, layers = 4, units=100, lr = 5e-3, fit_args = {}, **model_args):

    evidential_network = build_vanilla_keras_deep_evidential(
        train_data,
        layers,
        units
    )

    optimizer = KerasOptimizer(
        tf.keras.optimizers.Adam(lr),
        fit_args
    )

    deep_evidential = DeepEvidentialRegression(evidential_network, optimizer, **model_args)

    return deep_evidential

def plot_cubic(train_data:Dataset, test_data: Dataset, ood_predictions):

    fig, ax = plt.subplots(figsize=(14,10))

    ax.axvline(-4, color="grey", linestyle="dashed")
    ax.axvline(4, color="grey", linestyle="dashed")
    ax.plot(test_data.query_points, test_data.observations, color="red", linestyle="dashed")
    plot_scatter_with_var(test_data.query_points, ood_predictions[0], ood_predictions[1], ax=ax)
    ax.scatter(train_data.query_points, train_data.observations, color="tab:red", s=4, alpha = 0.6)
    ax.set_ylim(-150, 150)
    return fig

#%%

seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

with open("/home/clinton/Documents/bse/masters_thesis/trieste/notebooks/amini_cubic.pkl", "rb") as infile:
    x_train, y_train, x_test, y_test = pickle.load(infile)

x_train = tf.cast(x_train, dtype=tf.float64)
x_test = tf.cast(x_test, dtype=tf.float64)
y_train = tf.cast(y_train, dtype=tf.float64)
y_test = tf.cast(y_test, dtype=tf.float64)


train_data = Dataset(x_train, y_train)
test_data = Dataset(x_test, y_test)

# n=1000
# train_data = gen_cubic_dataset(n, -4, 4, True)
# test_data = gen_cubic_dataset(n, -7, 7, False)


#%%

fit_args = {
                "verbose": 0,
                "epochs": 5000,
                "batch_size": 128,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=200, restore_best_weights=True
                    )
                ],
            }

# deep_evidential = main_cubic(
#     train_data, 
#     fit_args=fit_args,
#     reg_weight=1e-2,
#     maxi_rate=0.,
#     verbose=1
# )
# #%%
# deep_evidential.optimize(train_data)
# predictions = deep_evidential.predict(train_data.query_points)
# error = tf.abs(train_data.observations - predictions[0])
# mean_error = tf.reduce_mean(error, axis=0)
# print(f"mean abs error {mean_error}")
# #%%
# ood_predictions = deep_evidential.predict(test_data.query_points)
# plot_cubic(train_data, test_data, ood_predictions)

# # #%%
# # #DIAGNOSIS
# # gamma, lamb, alpha, beta = tf.split(deep_evidential.model(tf.expand_dims(test_data.query_points, axis=-1))[0], 4, axis=-1)

# # names = ["gamma", "lambda", "alpha", "beta"]

# # for name, output in zip(names, [gamma, lamb, alpha, beta]):
# #     print(f"{name} has max: {np.max(output)} and min {np.min(output)}")

# # %%
# plt.plot(deep_evidential.model.history.history["loss"])
# # plt.ylim(0, 100)
# deep_evidential.model.history.history["loss"][-10:]

# # %%

# plt.plot(test_data.query_points,ood_predictions[1])
# # %%
# #Run the same cubic data over and over again to check the figs for
# #consistency

# %%

def simulate_cubic(
    prefix,
    sims = 30, 
    seed=1234, 
    refresh_seed=False,
    train_data = None,
    test_data = None, 
    refresh_data = False, 
    n=1000,
    **main_args
):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    models = []
    for i in range(sims):
        if refresh_seed:
            random.seed(i)
            np.random.seed(i)
            tf.random.set_seed(i)
        if refresh_data:
            train_data, test_data = gen_cubic_train_test(n)

        deep_evidential = main_cubic(
            train_data=train_data,
            **main_args
        )
        deep_evidential.optimize(train_data)
        ood_predictions = deep_evidential.predict(test_data.query_points)
        fig = plot_cubic(train_data, test_data, ood_predictions)
    
        models.append(deep_evidential)
        fig.savefig(f"/home/clinton/Documents/bse/masters_thesis/trieste/notebooks/de_figs/{prefix}_fig{i}.png", facecolor="white", transparent=False)

    return models

models = simulate_cubic(
    prefix="l2_reg_0_001",
    train_data = train_data, 
    test_data = test_data,
    fit_args = fit_args,
    reg_weight=1e-2,
    maxi_rate=0.,
    verbose=0
)

#%%