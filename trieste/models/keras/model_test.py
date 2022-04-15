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
    normal_inverse_gamma_negative_log_likelihood,
    normal_inverse_gamma_regularizer,
    deep_evidential_regression_loss
)

import numpy as np
from trieste.objectives import branin, BRANIN_SEARCH_SPACE
from trieste.objectives.utils import mk_observer
import tensorflow_probability as tfp

#%%


def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*tf.math.log(np.pi/v)  \
        - alpha*tf.math.log(twoBlambda)  \
        + (alpha+0.5) * tf.math.log(v*(y-gamma)**2 + twoBlambda)  \
        + tf.math.lgamma(alpha)  \
        - tf.math.lgamma(alpha+0.5)

    return tf.reduce_mean(nll)


def NIG_Reg(y, gamma, v, alpha, reduce=True):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = tf.abs(y-gamma)

    evi = 2*v+(alpha)
    reg = error*evi

    return tf.reduce_mean(reg)

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg

y_true1 = tf.constant([[1.]])
y_true2 = tf.constant([[1.8]])

test_point1 = tf.constant([[0.8, 0.2, 1.5, 0.3]])
test_point2 = tf.constant([[2.3, 0.5, 2.3, 0.7]])

y = tf.constant([[1.5], [3.], [4.2]])

evidential_output = tf.constant([
    [2.3, 1.1, 1.4, 0.2],
    [3.5, 2.5, 1.8, 0.9],
    [4.1, 10.2, 3.4, 1.2]
])

#%%

gamma, v, alpha, beta = tf.split(test_point1, 4, axis=-1)
print(NIG_NLL(y_true1, gamma, v, alpha, beta))
print(NIG_Reg(y_true1, gamma, v, alpha))

gamma, v, alpha, beta = tf.split(test_point2, 4, axis=-1)
print(NIG_NLL(y_true2, gamma, v, alpha, beta))
print(NIG_Reg(y_true2, gamma, v, alpha))



#%%
n=100
x = tf.expand_dims(tfp.distributions.Uniform(-3, 3).sample(n), axis=-1)

def paper_obj(x):
    n = len(x)
    errors = tfp.distributions.Normal(0, 0.02).sample(n)
    errors = tf.expand_dims(errors, axis=-1)
    y = tf.math.sin(3*x) / (3*x) + errors
    return y

y = paper_obj(x)


data = Dataset(x, y)

#%%
evidential_network = build_vanilla_keras_deep_evidential(
    data,
    3,
    300,
    evidence_activation = "softplus"
)

fit_args = {
                "verbose": 0,
                "epochs": 5000,
                "batch_size": 32,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=200, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="loss", patience=80, factor=0.1, verbose=1
                    )
                ],
            }

optimizer = KerasOptimizer(
    tf.keras.optimizers.Adam(0.001),
    fit_args,
    build_deep_evidential_regression_loss("NLL", coeff =1e-2)
)

deep_evidential = DeepEvidentialRegression(evidential_network, optimizer)

deep_evidential.optimize(data)

predictions = deep_evidential.predict(data.query_points)

error = tf.abs(data.observations - predictions[0])
mean_error = tf.reduce_mean(error, axis=0)
print(f"mean abs error {mean_error}")

# %%
plt.plot(deep_evidential.model.history.history["loss"])
# plt.ylim(0, 100)
deep_evidential.model.history.history["loss"][-30:]

# %%
fig, ax = plt.subplots()
ax.scatter(predictions[0], data.observations, color="black", label="predictions")
# %%

x_new = tf.cast(tf.expand_dims(tf.linspace(-10, 10, 1000), axis=-1), dtype=tf.float32)
y_true = paper_obj(x_new)
y_new, y_new_var = deep_evidential.predict(x_new)

y_upper = y_new + 1 * y_new_var**0.5
y_lower = y_new - 1 * y_new_var**0.5

fig2, ax2 = plt.subplots()
ax2.plot(x_new, y_true, color="red", label="True")
ax2.plot(x_new, y_new, color="blue", label="predicted")
ax2.plot(x_new, y_upper, color="grey", label="confidence bounds")
ax2.plot(x_new, y_lower, color="grey")
ax2.axvline(-3, color="grey", linestyle="dashed")
ax2.axvline(3, color="grey", linestyle="dashed")
ax2.set_ylim(-2, 2)

#%%

gamma, lamb, alpha, beta = tf.split(deep_evidential.model(data.query_points), 4, axis=-1)

names = ["gamma", "lambda", "alpha", "beta"]

for name, output in zip(names, [gamma, lamb, alpha, beta]):
    print(f"{name} has max: {np.max(output)} and min {np.min(output)}")
# %%

# %%
