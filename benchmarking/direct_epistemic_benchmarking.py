# %% 

import os
import tensorflow as tf

from benchmarking_utils import *
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

simul_args = common_simul_args

simul_args.update({
    "objective": [michal2, branin2],#, dropwave, eggholder, hartmann6],
    "num_initial_points": [1, 20],
    "acquisition": [("ei", EfficientGlobalOptimization()), ("ts", DiscreteThompsonSampling(2000, 4))],
    "num_steps": 20,
    "model": ("deup", deup_builder),
    "output_path": "deup_test",
    "ensemble_size": [5, 7],
    "num_hidden_layers": [3, 5],
    "units": [25, 50],
    "e_num_hidden_layers": [3, 5],
    "e_units": [64, 128, 256],
    "lr": 0.001,
    "seed": list(range(10))
})

multi_experiment(simul_args)
