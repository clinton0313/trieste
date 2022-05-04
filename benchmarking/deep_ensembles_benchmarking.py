# %% 

import os
import tensorflow as tf

from benchmarking_utils import (
    deepensemble_builder,
    multi_experiment,
    dropwave,
    eggholder,
    branin,
    michal2,
    hartmann6
)
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

simul_args = {
    "objective": [michal2, branin, dropwave, eggholder, hartmann6],
    "num_initial_points": [1, 20],
    "acquisition": [("ei", EfficientGlobalOptimization()), ("ts", DiscreteThompsonSampling(2000, 4))],
    "num_steps": [20],
    "model": [("de", deepensemble_builder)],
    "output_path": ["deep_ensemble_test"],
    "ensemble_size": [5, 7],
    "num_hidden_layers": [3, 5],
    "units": [25, 50],
    "plot": [False],
    "seed": list(range(10))
}

multi_experiment(simul_args)