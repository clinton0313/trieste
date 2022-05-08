import os
import tensorflow as tf

from benchmarking_utils import *
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def combine_args(simul_args_list: dict, common_args: dict) -> dict:
    '''Small helper function to combine simulation arguments'''
    for simul_args in simul_args_list:
        simul_args.update(common_args)
    return simul_args_list

OUTPUT_PATH = "parallel_benchmarking"

common_args = {
    "objective": [michal2, branin],#, dropwave, eggholder, hartmann6],
    "num_initial_points": [1, 20],
    "acquisition": [("ei", EfficientGlobalOptimization()), ("ts", DiscreteThompsonSampling(2000, 4))],
    "num_steps": 20,
    "predict_interval": 3,
    "plot": False,
    "report_predictions": True,
    "overwrite": False,
    "grid_density": 20,
    "metadata": "",
    "seed": list(range(10))
}

der_simul_args = {
    "model": ("new_der_log", der_builder),
    "output_path": os.path.join(OUTPUT_PATH, "der"),
    "num_hidden_layers": [2, 4],
    "units": [50, 100],
    "reg_weight": [1e-3, 1e-4],
    "maxi_rate": [0, 1e-2],
    "lr": 0.001,
}

gpr_simul_args = {
    "acquisition": [("ei", EfficientGlobalOptimization()), ("ts", DiscreteThompsonSampling(2000, 4))],
    "num_steps": 20,
    "model": ("gpr", gpr_builder),
    "output_path": os.path.join(OUTPUT_PATH, "gpr"),
}

de_simul_args = {
    "model": ("de", deepensemble_builder),
    "output_path": os.path.join(OUTPUT_PATH, "de"),
    "ensemble_size": [5, 7],
    "num_hidden_layers": [3, 5],
    "units": [25, 50],
}

mc_simul_args = {
    "model": ("mc", mcdropout_builder),
    "output_path": os.path.join(OUTPUT_PATH, "mc"),
    "num_hidden_layers": [3, 5],
    "units": [25, 50, 100],
    "rate": [0.1, 0.2, 0.3],
    "num_passes": 100,
    "lr": 0.001,
}

deup_simul_args = {
    "model": ("deup", deup_builder),
    "output_path": os.path.join(OUTPUT_PATH, "deup"),
    "ensemble_size": [5, 7],
    "num_hidden_layers": [3, 5],
    "units": [25, 50],
    "e_num_hidden_layers": [3, 5],
    "e_units": [64, 128, 256],
    "lr": 0.001,
}

n_jobs = 2
verbose = 5

#Each dictionary of args is independently crosses all of its arguments
all_args = [
    der_simul_args,
    de_simul_args,
    gpr_simul_args,
    mc_simul_args,
    deup_simul_args
]

if __name__ == "__main__":
    os.mkdir(OUTPUT_PATH, exists_ok=True)
    combine_args(all_args, common_args)
    parallel_experiments(all_args, n_jobs = n_jobs, verbose = verbose)