#%%
import joblib
import os
import tensorflow as tf

from benchmarking_utils import *
from functools import partial
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

OUTPUT_PATH = "test"

common_args = {
    "objective": [
#         branin2,
#         michal2,
#         goldstein2,
#         hartmann3,
#         shekel4
        dropw2,
        eggho2,
        rosenbrock4,
        hartmann6,
        ackley5,
    ],
    "num_initial_points": 10,
    "acquisition":  [
        ("ei", EfficientGlobalOptimization, {}), 
        ("ts", DiscreteThompsonSampling,{"num_search_space_samples": 1000, "num_query_points": 4})
        #num_search_space_samples for discrete thompson sampling is multiplied by number of search space dimensions
    ],
    "num_steps": 500, #10 * number of search space dimensions
    "predict_interval": 4,
    "plot": False,
    "report_predictions": True,
    "overwrite": False,
    "tolerance": 1e-3,
    "grid_density": 10000,
    "metadata": "",
    "seed": [x for x in range(20)],
    "sample_seed": 42,
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
    "acquisition": ("ei", EfficientGlobalOptimization, {}),
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
    "num_passes": 50,
    "lr": 0.001,
}

#UNKNOWN ARGS
deup_simul_args = {
    "model": ("deup", deup_builder),
    "output_path": os.path.join(OUTPUT_PATH, "deup"),
#     "ensemble_size": 7,
#     "num_hidden_layers": 5,
    "units": 50,
    "e_num_hidden_layers": 5,
    "e_units": 128,
    "init_buffer_iters": 2,
    "lr": 0.001,
}

svgp_simul_args = {
    "model": ("svgp", svgp_builder),
    "num_inducing_points": 500,
    "output_path": os.path.join(OUTPUT_PATH, "svgp"),
}

n_jobs = joblib.cpu_count()
verbose = 50 #From 1 to 50 

#Each dictionary of args is independently crosses all of its arguments
all_args = [
#     der_simul_args,
    # de_simul_args,
    mc_simul_args,
    # gpr_simul_args,
    deup_simul_args,
    svgp_simul_args
]
#%%
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    all_args = list(map(partial(combine_args, common_args=common_args), all_args))
    parallel_experiments(all_args, n_jobs = n_jobs, verbose = verbose)
