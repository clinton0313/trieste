# %%
# %load_ext autoreload
# %autoreload 2
import tensorflow as tf
import trieste
import matplotlib.pyplot as plt
from trieste.models.optimizer import KerasOptimizer
tf.get_logger().setLevel("ERROR")

from trieste.data import Dataset
from trieste.models.keras import (
    DeepEvidentialNetwork, 
    DeepEvidentialRegression,
    build_vanilla_keras_deep_evidential, 
    get_tensor_spec_from_data
)

from trieste.models.keras.utils import (
    build_deep_evidential_regression_loss
)

import numpy as np
import tensorflow_probability as tfp
tf.keras.backend.set_floatx("float64")
#%%


n=1000
x = tf.expand_dims(tfp.distributions.Uniform(-3, 3).sample(n), axis=-1)

def cubic(x, noise=True):
    y = tf.pow(x, 3)
    if noise:
        sigma = np.ones_like(x) * 3
        y += np.random.normal(0, sigma)
    return y

def gen_cubic_dataset(n, min, max, noise=True):
    x = tf.linspace(min, max, n)
    x = tf.cast(tf.expand_dims(tf.sort(x), axis=-1), dtype=tf.float64)
    y = cubic(x, noise)
    return Dataset(x,y)

cubic_data = gen_cubic_dataset(n, -4, 4, True)


#%%
evidential_network = build_vanilla_keras_deep_evidential(
    cubic_data,
    3,
    100
)

fit_args = {
                "verbose": 0,
                "epochs": 5000,
                "batch_size": 128,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=1000, restore_best_weights=True
                    )
                ],
            }

optimizer = KerasOptimizer(
    tf.keras.optimizers.Adam(5e-3),
    fit_args,
    build_deep_evidential_regression_loss(coeff =1e-2)
)

deep_evidential = DeepEvidentialRegression(evidential_network, optimizer)

deep_evidential.optimize(cubic_data)

predictions = deep_evidential.predict(cubic_data.query_points)

error = tf.abs(cubic_data.observations - predictions[0])
mean_error = tf.reduce_mean(error, axis=0)
print(f"mean abs error {mean_error}")

# %%

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(cubic_data.query_points, cubic_data.observations, color="red", s=1)

def plot_scatter_with_var(query_points, y_pred, y_var, ax, n_stds=3, max_alpha = 0.7):
    x = tf.squeeze(query_points)
    y = tf.squeeze(y_pred)
    std = tf.squeeze(y_var**0.5) #needs a square root missing...
    ax.plot(x, y, color="black", label="predictions")
    for k in range(1, n_stds + 1):
        upper_std = y + k * std
        lower_std = y - k * std
        ax.fill_between(x, upper_std, lower_std, alpha = max_alpha/k, color="tab:blue")
# %%

ood_x = tf.linspace(-7, 7, 1000)
ood_y = cubic(ood_x, noise=False)
ood_predictions = deep_evidential.predict(ood_x)
ax.axvline(-4, color="grey", linestyle="dashed")
ax.axvline(4, color="grey", linestyle="dashed")
ax.plot(ood_x, ood_y, color="red", linestyle="dashed")
plot_scatter_with_var(ood_x, ood_predictions[0], ood_predictions[1], ax=ax)
ax.set_ylim(-150, 150)
fig
#%%

loss_fn = build_deep_evidential_regression_loss(coeff=1e-2)
evidential_output = deep_evidential.model(cubic_data.query_points)
loss = loss_fn(cubic_data.observations, evidential_output)
g, v, a, b = tf.split(evidential_output, 4, axis=-1)

#%%
#DIAGNOSIS
gamma, lamb, alpha, beta = tf.split(deep_evidential.model(tf.expand_dims(ood_x, axis=-1)), 4, axis=-1)

names = ["gamma", "lambda", "alpha", "beta"]

for name, output in zip(names, [gamma, lamb, alpha, beta]):
    print(f"{name} has max: {np.max(output)} and min {np.min(output)}")

# %%
plt.plot(deep_evidential.model.history.history["loss"])
# plt.ylim(0, 100)
deep_evidential.model.history.history["loss"][-10:]

# %%
plt.plot(ood_x,ood_predictions[1])
# %%
#Run the same cubic data over and over again to check the figs for
#consistency

def main_cubic(cubic_data, coeff=1e-2):
    evidential_network = build_vanilla_keras_deep_evidential(
        cubic_data,
        3,
        100
    )

    fit_args = {
                    "verbose": 0,
                    "epochs": 5000,
                    "batch_size": 128,
                    "callbacks": [
                        tf.keras.callbacks.EarlyStopping(
                            monitor="loss", patience=1000, restore_best_weights=True
                        )
                    ],
                }

    optimizer = KerasOptimizer(
        tf.keras.optimizers.Adam(5e-3),
        fit_args,
        build_deep_evidential_regression_loss(coeff =coeff)
    )

    deep_evidential = DeepEvidentialRegression(evidential_network, optimizer)
    deep_evidential.optimize(cubic_data)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(cubic_data.query_points, cubic_data.observations, color="red", s=1)

    ood_x = tf.linspace(-7, 7, 1000)
    ood_y = cubic(ood_x, noise=False)
    ood_predictions = deep_evidential.predict(ood_x)
    ax.axvline(-4, color="grey", linestyle="dashed")
    ax.axvline(4, color="grey", linestyle="dashed")
    ax.plot(ood_x, ood_y, color="red", linestyle="dashed")
    plot_scatter_with_var(ood_x, ood_predictions[0], ood_predictions[1], ax=ax)
    ax.set_ylim(-150, 150)
    return fig

#%%
sims = 10
figs = [main_cubic(cubic_data) for _ in range(sims)]
# %%

f = main_cubic(cubic_data, coeff = 1e-2)
# %%
