#%%
import itertools
import os
import pickle
import pytest
import tensorflow as tf
from time import time
from tqdm import tqdm

from tests.util.misc import branin_dataset, random_seed
from trieste.models.keras import build_vanilla_keras_mcdropout
from trieste.models.keras.models import DeepDropout
from trieste.models.optimizer import KerasOptimizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%


def integration_test(passes, lr, params):
    example_data = branin_dataset(1000)
    deep_dropout = build_vanilla_keras_mcdropout(
        example_data, 
        **params
    )

    optimizer = tf.keras.optimizers.Adam(lr)
    fit_args = {
        "batch_size": 16, 
        "epochs": 1000,
        "callbacks": [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
        "verbose": 0
    }

    model = DeepDropout(
        deep_dropout,
        KerasOptimizer(optimizer, fit_args),
        passes=passes
    )

    start = time()
    model.optimize(example_data)

    predicted_means, predicted_variance = model.predict(example_data.query_points)
    elapsed = time() - start

    mae = tf.reduce_mean(tf.abs(predicted_means - example_data.observations)).numpy()

    return mae, elapsed
# %%

def cv(num_passes, lrs, param_dict, savefile, func=integration_test):
    
    try:
        with open(savefile, "rb") as infile:
            results = pickle.load(infile)
        tested_params = [r["params"] for r in results]
    except FileNotFoundError:
        results = []
        tested_params = []

    keys, values = zip(*param_dict.items())
    for lr in lrs:
        for passes in num_passes:
            for v in tqdm(itertools.product(*values)):
                params = dict(zip(keys, v))
                if params in tested_params:
                    continue
                mae, elapsed = func(passes, lr, params)
                r = {
                    "mae": mae,
                    "time": elapsed,
                    "passes": passes,
                    "lr": lr,
                    "params": params
                }
                results.append(r)
                with open(savefile, "wb") as outfile:
                    pickle.dump(results, outfile)
                tqdm.write(f"Saved results {r}")

num_passes = [100, 300, 500]
lrs = [0.01, 0.001, 0.0001]
param_dict = {
    "num_hidden_layers":[5, 10, 15],
    "units":[100, 300, 500],
    "activation": ["relu"],
    "rate": [0.1, 0.3, 0.5],
    "dropout": ["standard", "dropconnect"]
}

cv(num_passes=num_passes, lrs=lrs, param_dict=param_dict, savefile="results.pkl")

#%%