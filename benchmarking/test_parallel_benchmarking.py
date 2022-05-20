#%%
import joblib
import os
import tensorflow as tf

from benchmarking_utils import *
from functools import partial
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization, RandomSampling

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

OUTPUT_PATH = "test_parallel_benchmarking"

common_args = {
    "objective": REGULAR_OBJECTIVES,
    "num_initial_points": 10,
    "acquisition":  [
        ("rand", RandomSampling, {})
        #num_search_space_samples for discrete thompson sampling is multiplied by number of search space dimensions
        ],
    "num_steps": 2000,
    "predict_interval": 4,
    "plot": False,
    "report_predictions": True,
    "overwrite": False,
    "tolerance": 1e-3,
    "grid_density": 20,
    "metadata": "",
    "seed": list(range(30)),
    "sample_seed": 42,
    "verbose_output": False
}

random_simul_args = {
    "model": ("random", dummy_builder),
    "output_path": os.path.join(OUTPUT_PATH, "random"),
}


n_jobs = 2
verbose = 50 #From 1 to 50 

#Each dictionary of args is independently crosses all of its arguments
all_args = [
    random_simul_args
]
#%%
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    all_args = list(map(partial(combine_args, common_args=common_args), all_args))
    parallel_experiments(all_args, n_jobs = n_jobs, verbose = verbose)