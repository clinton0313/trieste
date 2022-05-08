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
    "objective": NOISY_OBJECTIVES,
    "num_initial_points": 20,
    "acquisition": [("ei", EfficientGlobalOptimization()), ("ts", DiscreteThompsonSampling(2000, 4))],
    "num_steps": "infer", #infer hardcodes to num_dimensions * 10
    "predict_interval": 4,
    "plot": False,
    "report_predictions": True,
    "overwrite": False,
    "grid_density": 20,
    "metadata": "",
    "seed": list(range(20)),
    "verbose_output": False
}

#Primary: num_layers: 4, units: 100, reg_weight: 0.001, maxi_rate: 0.01, lr: 0.001 (standard)
#Secondary: num_layers: 4, units: 100, reg_weight: 0.001, maxi_rate: 0, lr: 0.001 (standard)

#Primary: num_layers: 4, units: 100, reg_weight: 0.001, maxi_rate: 0.01, lr: 0.001 (log)
#Secondary: num_layers: 4, units: 50, reg_weight: 0.001, maxi_rate: 0, lr: 0.001 (log)
der_simul_args = {
    "model": ("new_der_log", der_builder),
    "output_path": os.path.join(OUTPUT_PATH, "der"),
    "num_hidden_layers": 4,
    "units": 100,
    "reg_weight": 1e-3,
    "maxi_rate": 1e-2,
    "lr": 0.001,
}

gpr_simul_args = {
    "model": ("gpr", gpr_builder),
    "output_path": os.path.join(OUTPUT_PATH, "gpr"),
}

#Primary: ensemble_size: 7, num_layers: 5, units: 50
#Secondary: ensemble_size: 5, num_layers: 3, units: 25
de_simul_args = {
    "model": ("de", deepensemble_builder),
    "output_path": os.path.join(OUTPUT_PATH, "de"),
    "ensemble_size": 7,
    "num_hidden_layers": 5,
    "units": 50,
}

#Primary: num_layers: 5, units: 100, rate: 0.1, passes: 100, lr: 0.001
#Secondary: num_layers: 3, units: 100, rate: 0.1, passes: 100, lr: 0.001
mc_simul_args = {
    "model": ("mc", mcdropout_builder),
    "output_path": os.path.join(OUTPUT_PATH, "mc"),
    "num_hidden_layers": 5,
    "units": 100,
    "rate": 0.1,
    "num_passes": 100,
    "lr": 0.001,
}

#UNKNOWN ARGS
# deup_simul_args = {
#     "model": ("deup", deup_builder),
#     "output_path": os.path.join(OUTPUT_PATH, "deup"),
#     "ensemble_size": [5, 7],
#     "num_hidden_layers": [3, 5],
#     "units": [25, 50],
#     "e_num_hidden_layers": [3, 5],
#     "e_units": [64, 128, 256],
#     "lr": 0.001,
# }

n_jobs = 2
verbose = 10 #From 1 to 50 

#Each dictionary of args is independently crosses all of its arguments
all_args = [
    der_simul_args,
    de_simul_args,
    mc_simul_args,
    gpr_simul_args,
    # deup_simul_args
]

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    combine_args(all_args, common_args)
    parallel_experiments(all_args, n_jobs = n_jobs, verbose = verbose)