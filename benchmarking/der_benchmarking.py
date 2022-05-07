# %% 

import os
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
from benchmarking_utils import (
    deepensemble_builder,
    der_builder,
    mcdropconnect_builder,
    mcdropout_builder,
    multi_experiment,
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
    "objective": [branin],
    "num_initial_points": [20],
    "acquisition": [ ("ei", EfficientGlobalOptimization())],
    "num_steps": 9,
    "predict_interval": 3,
    "model": [("new_der_log", der_builder)],
    "output_path": "new_der_test",
    "num_hidden_layers": [2, 4],
    "units": [50, 100],
    "reg_weight": [1e-3, 1e-4],
    "maxi_rate": [0, 1e-2],
    "lr": 0.001,
    "plot": False,
    "report_predictions": True,
    "seed": list(range(3)),
    "grid_density": 20,
    "metadata": "",
    "overwrite": False
}

multi_experiment(simul_args)
# with ThreadPoolExecutor(max_workers=2) as executor:
#     future = executor.submit(multi_experiment, simul_args)
#     print(future.result())