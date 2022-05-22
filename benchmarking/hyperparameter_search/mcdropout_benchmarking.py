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
    "acquisition":  [
        ("ei", EfficientGlobalOptimization, {}), 
        ("ts", DiscreteThompsonSampling,{"num_search_space_samples": "infer", "num_query_points": 4})
    ],
    "num_steps": 20,
    "model": [("mc", mcdropout_builder)],
    "output_path": ["mcdropout_test"],
    "num_hidden_layers": [3, 5],
    "units": [25, 50, 100],
    "rate": [0.1, 0.2, 0.3],
    "num_passes": 100,
    "lr": 0.001,
    "seed": list(range(10))
})

multi_experiment(simul_args)