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
    "objective": branin,
    "num_initial_points": 20,
    "acquisition": [ ("ei", EfficientGlobalOptimization())],
    "num_steps": 9,
    "predict_interval": 3,
    "model": ("new_der_log", der_builder),
    "output_path": "new_der_test",
    "num_hidden_layers": [2, 4],
    "units": [50, 100],
    "reg_weight": [1e-3, 1e-4],
    "maxi_rate": [0, 1e-2],
    "lr": 0.001,
    "seed": list(range(10))
})

multi_experiment(simul_args)
