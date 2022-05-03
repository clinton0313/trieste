# %% 

import os
import tensorflow as tf

from benchmarking_utils import (
    deepensemble_builder,
    simulate_experiment, 
    multi_experiment,
    branin,
    michal2
)
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
#MAKE OBJECTIVE IF NOT USING PREDEFINED OBJECTIVES IN UTILS

michal2 = ("michal2", michalewicz_2, MICHALEWICZ_2_SEARCH_SPACE, MICHALEWICZ_2_MINIMUM, MICHALEWICZ_2_MINIMIZER)
branin = ("scaled_branin", scaled_branin, BRANIN_SEARCH_SPACE, SCALED_BRANIN_MINIMUM, BRANIN_MINIMIZERS)

#MAKE BUILDER IF NOT USING PREDEFINED BUILDERS IN UTILS

def build_der(data, num_hidden_layers, units, reg_weight, maxi_rate):
    network = build_vanilla_keras_deep_evidential(
        data=data,
        num_hidden_layers=num_hidden_layers,
        units=units,
        )
    model = DeepEvidentialRegression(network, reg_weight=reg_weight, maxi_rate=maxi_rate)
    return model

#DEFINE PREFIXES FOR SAVEFILES IF NOT USING GLOBAL DEFAULT

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
    acquisition=("ts", DiscreteThompsonSampling(2000, 4)),
    num_steps=25,
    model=("der", build_der),
    output_path="der_test",
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

simul_args = {
    "objective": [michal2],
    "num_initial_points": [1],
    "acquisition": [("ei", EfficientGlobalOptimization())],
    "num_steps": [25],
    "model": [("der", deepensemble_builder)],
    "output_path": ["deep_ensemble_test"],
    "ensemble_size": [2, 3],
    "num_hidden_layers": [2, 3],
    "units": [25]
}

multi_experiment(simul_args)