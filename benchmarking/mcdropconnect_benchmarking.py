# %% 

import os
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from benchmarking_utils import (
    deepensemble_builder,
    mcdropconnect_builder,
    mcdropout_builder,
    multi_experiment,
    branin,
    michal2,
    dropwave,
    eggholder,
    hartmann6
)
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

simul_args = {
    "objective": [michal2, branin],#, dropwave, eggholder, hartmann6],
    "num_initial_points": [1, 20],
    "acquisition": [("ei", EfficientGlobalOptimization()), ("ts", DiscreteThompsonSampling(2000, 4))],
    "num_steps": [20],
    "model": [("mcdc", mcdropconnect_builder)],
    "output_path": ["mcdropconnect_test"],
    "num_hidden_layers": [3, 5],
    "units": [25, 50, 100],
    "rate": [0.1, 0.2, 0.3],
    "num_passes": [100],
    "lr": [0.001],
    "plot": [False],
    "seed": list(range(10))
}

with ThreadPoolExecutor(max_workers=20) as executor:
    future = executor.submit(multi_experiment, simul_args)
    print(future.result())
# %%
