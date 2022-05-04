# %% 

import os
import tensorflow as tf

from benchmarking_utils import (
    deepensemble_builder,
    mcdropconnect_builder,
    mcdropout_builder,
    multi_experiment,
    branin,
    michal2
)
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

simul_args = {
    "objective": [michal2, branin],
    "num_initial_points": [1, 20],
    "acquisition": [("ei", EfficientGlobalOptimization()), ("ts", DiscreteThompsonSampling(2000, 4))],
    "num_steps": [20],
    "model": [("mc", mcdropout_builder), ("mcdc", mcdropconnect_builder)],
    "output_path": ["mcdropout_test"],
    "num_hidden_layers": [2, 3, 5],
    "units": [25, 50, 100],
    "rate": [0.1, 0.2, 0.3],
    "num_passes": [100, 200],
    "lr": [0.001, 0.01],
    "plot": [False],
    "seed": list(range(10))
}

multi_experiment(simul_args)