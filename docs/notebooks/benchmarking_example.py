# %% 

import os
import tensorflow as tf
import trieste

from benchmarking_utils import simulate_experiment, multi_experiment
from trieste.acquisition.rule import DiscreteThompsonSampling, EfficientGlobalOptimization
from trieste.models.keras import (
    DeepEvidentialRegression,
    build_vanilla_keras_deep_evidential, 
)
from trieste.objectives import (
    michalewicz_2,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_2_SEARCH_SPACE,
    MICHALEWICZ_2_MINIMIZER,
    scaled_branin,
    SCALED_BRANIN_MINIMUM,
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")
os.chdir(os.path.dirname(os.path.realpath(__file__)))


#%%
#MAKE OBJECTIVE

michal2 = ("michal2", michalewicz_2, MICHALEWICZ_2_SEARCH_SPACE, MICHALEWICZ_2_MINIMUM, MICHALEWICZ_2_MINIMIZER)
branin = ("scaled_branin", scaled_branin, BRANIN_SEARCH_SPACE, SCALED_BRANIN_MINIMUM, BRANIN_MINIMIZERS)

#MAKE BUILDER

def build_der(data, num_hidden_layers, units, reg_weight, maxi_rate):
    network = build_vanilla_keras_deep_evidential(
        data=data,
        num_hidden_layers=num_hidden_layers,
        units=units,
        )
    model = DeepEvidentialRegression(network, reg_weight=reg_weight, maxi_rate=maxi_rate)
    return model

#DEFINE PREFIXES FOR SAVEFILES

save_title_prefixes = {
    "num_hidden_layers": "L",
    "units": "n",
    "reg_weight": "r",
    "maxi_rate": "m"
}

#RUN EXPERIMENT

simulate_experiment(
    objective=branin,
    num_initial_points=1,
    acquisition_rule=DiscreteThompsonSampling(2000, 4),
    acquisition_name="ts",
    num_steps=25,
    model_builder=build_der,
    model_name="der",
    output_path="test",
    save_title_prefixes=save_title_prefixes,
    plot=True,
    seed=0,
    num_hidden_layers=2,
    units=25,
    reg_weight=1e-4,
    maxi_rate=1e-2
)

# %%

#CROSS MULTIPLE EXPERIMENTS LIKE THIS:

# simul_args = {
#     "objective": [branin],
#     "num_initial_points": [1],
#     "acquisition_rule": [DiscreteThompsonSampling(2000, 4)],
#     "acquisition_name": ["ts"],
#     "num_steps": [25],
#     "model_builder": [build_der],
#     "model_name": ["der"],
#     "output_path": ["test"],
#     "save_title_prefixes": [save_title_prefixes],
#     "plot": [True],
#     "seed": [0],
#     "num_hidden_layers": [2, 3],
#     "units": [10, 20],
#     "reg_weight": [1e-4],
#     "maxi_rate": [1e-2],
# }

# multi_experiment(simul_args)