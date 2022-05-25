#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from trieste.objectives.single_objectives import HARTMANN_3_SEARCH_SPACE, hartmann_3

os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%
n = 40
ss = HARTMANN_3_SEARCH_SPACE

coords = tf.linspace(ss.lower, ss.upper, n)
x, y, z = tf.split(coords, 3, axis=1)

xx, yy, zz = tf.meshgrid(x, y, z)

x1 = tf.reshape(xx, -1)
x2 = tf.reshape(yy, -1)
x3 = tf.reshape(zz, -1)

query_points = tf.stack([x1, x2, x3], axis = 1)
observations = hartmann_3(query_points)
# %%
f = plt.figure(figsize=(12, 12))

a = plt.axes(projection="3d")
obs = observations.numpy().squeeze()
obs = (obs  - min(obs) ) / (max(obs) - min(obs))
viridis = matplotlib.cm.get_cmap('viridis', 256)

a.scatter(x1, x2, tf.expand_dims(x3, axis=1), c = viridis(obs), alpha = 0.8)
a.set_title("Hartmann-3")
f.savefig(os.path.join("function_figs", "hartmann3.png"), facecolor="white", transparent=False)
# %%
