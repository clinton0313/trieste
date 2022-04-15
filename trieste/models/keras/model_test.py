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


n=100
x = tf.expand_dims(tfp.distributions.Uniform(-3, 3).sample(n), axis=-1)

def cubic(x, noise=True):
    y = tf.pow(x, 3)
    if noise:
        y += np.random.normal(0, 3**0.5, len(x)).reshape((-1,1))
    return y

def gen_cubic_dataset(n, min, max, noise=True):
    x = tfp.distributions.Uniform(min, max).sample(n)
    x = tf.cast(tf.expand_dims(tf.sort(x), axis=-1), dtype=tf.float64)
    y = cubic(x, noise)
    return Dataset(x,y)

cubic_data = gen_cubic_dataset(1000, -4, 4)


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
                        monitor="loss", patience=200, restore_best_weights=True
                    ),
                    # tf.keras.callbacks.ReduceLROnPlateau(
                    #     monitor="loss", patience=80, factor=0.1, verbose=1
                    # )
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

def plot_scatter_with_var(query_points, predictions, ax, n_stds=3, max_alpha = 0.7):
    x = tf.squeeze(query_points)
    y = tf.squeeze(predictions[0])
    std = tf.squeeze(predictions[1]) ** 0.5
    ax.plot(x, y, color="black", label="predictions")
    for k in range(1, n_stds + 1):
        upper_std = y + k * std
        lower_std = y - k * std
        ax.fill_between(x, upper_std, lower_std, alpha = max_alpha/k, color="tab:blue")
# %%

ood_x = tf.linspace(-6, 6, 1000)
ood_y = cubic(ood_x, noise=False)
ood_predictions = deep_evidential.predict(ood_x)

ax.plot(ood_x, ood_y, color="red", linestyle="dashed")
plot_scatter_with_var(ood_x, ood_predictions, ax=ax)

#%%
#DIAGNOSIS
gamma, lamb, alpha, beta = tf.split(deep_evidential.model(tf.expand_dims(ood_x, axis=-1)), 4, axis=-1)

names = ["gamma", "lambda", "alpha", "beta"]

for name, output in zip(names, [gamma, lamb, alpha, beta]):
    print(f"{name} has max: {np.max(output)} and min {np.min(output)}")

# %%
plt.plot(deep_evidential.model.history.history["loss"])
# plt.ylim(0, 100)
deep_evidential.model.history.history["loss"][-30:]

# %%
