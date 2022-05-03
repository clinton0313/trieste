# %% 

import os
import tensorflow as tf

from benchmarking_utils import (
    deepensemble_builder,
    multi_experiment,
    branin,
    michal2
)
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#CROSS MULTIPLE EXPERIMENTS LIKE THIS:

simul_args = {
    "objective": [michal2, branin],
    "num_initial_points": [1],
    "acquisition_rule": [EfficientGlobalOptimization()],
    "acquisition_name": ["ei"],
    "num_steps": [25],
    "model_builder": [deepensemble_builder],
    "model_name": ["der"],
    "output_path": ["deep_ensemble_test"],
    "ensemble_size": [5],
    "num_hidden_layers": [2],
    "units": [25],
    "seeds": list(range(10))
}

multi_experiment(simul_args)