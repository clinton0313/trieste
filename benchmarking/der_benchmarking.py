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
    "objective": [michal2, branin2],
    "num_initial_points": 20,
    "acquisition":  [
        ("ei", EfficientGlobalOptimization, {}), 
        ("ts", DiscreteThompsonSampling,{"num_search_space_samples": "infer", "num_query_points": 4})
    ],
    "num_steps": 20,
    "predict_interval": 3,
    "model": ("der_log_ei", der_builder),
    "output_path": "der_test",
    "num_hidden_layers": [2, 4],
    "units": [50, 100],
    "reg_weight": [1e-3, 1e-4],
    "maxi_rate": [0, 1e-2],
    "lr": 0.001,
    "report_predictions": False,
    "seed": list(range(10))
})

parallel_experiments(simul_args, n_jobs=2)
