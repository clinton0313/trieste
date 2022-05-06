# %% 

import os
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
from benchmarking_utils import (
    gpr_builder,
    multi_experiment,
    branin,
    michal2,
    dropwave,
    eggholder,
    hartmann6
)
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization
from trieste.objectives.single_objectives import BRANIN_SEARCH_SPACE, MICHALEWICZ_2_SEARCH_SPACE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

simul_args = {
    "objective": [branin, michal2], #, dropwave, eggholder, hartmann6],
    "num_initial_points": [2],
    "acquisition": [("ei", EfficientGlobalOptimization()), ("ts", DiscreteThompsonSampling(2000, 4))],
    "num_steps": [25],
    "model": [("gpr", gpr_builder)],
    "output_path": ["gpr_test"],
    "seed": list(range(10))
}

with ThreadPoolExecutor(max_workers=20) as executor:
    future = executor.submit(multi_experiment, simul_args)
    print(future.result())