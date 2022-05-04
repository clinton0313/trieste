import gpflow
import itertools
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow_probability as tfp 
import time
import trieste

from tqdm import tqdm
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.models.gpflow.builders import NUM_INDUCING_POINTS_PER_DIM
from trieste.models.keras import (
    DeepEvidentialRegression,
    MonteCarloDropout,
    DeepEnsemble,
    build_vanilla_keras_ensemble,
    build_vanilla_keras_mcdropout,
    build_vanilla_keras_deep_evidential, 
)
from trieste.models.keras.architectures import DropConnectNetwork
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
from trieste.objectives.single_objectives import HARTMANN_3_SEARCH_SPACE, HARTMANN_6_MINIMIZER, HARTMANN_6_MINIMUM, hartmann_6
from trieste.objectives.utils import mk_observer
from trieste.models.optimizer import KerasOptimizer
from typing import Callable, Tuple
from util.plotting_plotly import (
    plot_model_predictions_plotly, 
    plot_function_plotly,
    add_bo_points_plotly
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.keras.backend.set_floatx("float64")

#%%

global_save_title_prefixes = {
    "num_hidden_layers": "L",
    "units": "n",
    "reg_weight": "rw",
    "maxi_rate": "m",
    "rate": "dr",
    "lr": "lr",
    "ensemble_size": "es"
}

# OBJECTIVES

michal2 = ("michal2", michalewicz_2, MICHALEWICZ_2_SEARCH_SPACE, MICHALEWICZ_2_MINIMUM, MICHALEWICZ_2_MINIMIZER)
branin = ("scaled_branin", scaled_branin, BRANIN_SEARCH_SPACE, SCALED_BRANIN_MINIMUM, BRANIN_MINIMIZERS)
hartmann6 = ("hartmann6", hartmann_6, HARTMANN_3_SEARCH_SPACE, HARTMANN_6_MINIMUM, HARTMANN_6_MINIMIZER)

# MODEL BUILDERS

def der_builder(data, num_hidden_layers, units, reg_weight, maxi_rate, lr):
    network = build_vanilla_keras_deep_evidential(
        data=data,
        num_hidden_layers=num_hidden_layers,
        units=units,
        )
    model = DeepEvidentialRegression(
        network, 
        optimizer=KerasOptimizer(tf.optimizers.Adam(lr)),
        reg_weight=reg_weight, 
        maxi_rate=maxi_rate
    )
    return model

def mcdropout_builder(data, num_hidden_layers, units, rate, num_passes, lr):
    network = build_vanilla_keras_mcdropout(data, num_hidden_layers, units, rate=rate)
    model = MonteCarloDropout(network, KerasOptimizer(tf.optimizers.Adam(lr)), num_passes=num_passes)
    return model

def mcdropconnect_builder(data, num_hidden_layers, units, rate, num_passes, lr):
    network = build_vanilla_keras_mcdropout(data, num_hidden_layers, units, rate=rate, dropout_network=DropConnectNetwork)
    model = MonteCarloDropout(network, KerasOptimizer(tf.optimizers.Adam(lr)), num_passes=num_passes)
    return model

def deepensemble_builder(data, ensemble_size, num_hidden_layers, units):
    network = build_vanilla_keras_ensemble(data, ensemble_size, num_hidden_layers, units)
    model = DeepEnsemble(network)
    return model


def gpr_builder(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.cast(-2.0, dtype=tf.float64), prior_scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr, num_kernel_samples=100)

# SIMULATOR
def parse_rate(rate:float)-> str:
    '''Helpful function taht parses floats into scientific notation strings'''
    if rate > 10 or rate < -10:
        out = str(rate)
        if out == "0":
            return out
        elif out.count("e") == 1:
            out = out.replace("-", "").replace("0", "")
            return out
        else:
            zeros = out.count("0")
            out = out.replace("0", "").replace(".", "")
            return out + "e" + str(zeros)
    else:
        out = str(round(rate, 2))
        out = out.replace(".", "_")
        return out

def simulate_experiment(
    objective: Tuple, 
    num_initial_points: int,
    acquisition: Tuple[str, trieste.acquisition.rule.AcquisitionRule],
    num_steps: int,
    model: Tuple[str, Callable],
    output_path: str,
    save_title_prefixes: dict = global_save_title_prefixes,
    plot: bool = True,
    seed: int = 0,
    **model_args
):
    """
    :param objective: Tuple of (objective_name, function, search_space, minimum, minimizer)
    :param num_initial_points: Number of initial query points.
    :param acquisition: Tuple of (acquisition_name, instantiated Acquisition rule)
    :param num_steps: Number of bayesian optimization steps.
    :param model: Tuple of (model_name, model_builder). Model_builder is a function that accepts
        arguments: initial_data, and **model_args and returns a model. 
    :param output_path: Path to save figs and log_file
    :param save_title_prefixes: Dictionary of {model_arg: prefix} used for prefixing the save_title. For
        example: {'num_hidden_layers': 'hl'}. Will filter out unused prefixes hence defaults to 
        global_save_title_prefixes where we can add prefixes for all models. 
    :param plot: True or False, whether to generate plots or not. 
    :param seed: Seed.
    """

    #Set seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    #Unpack objective tuple:
    acquisition_name, acquisition_rule = acquisition
    model_name, model_builder = model
    objective_name, function, search_space, minimum, minimizer = objective

    #Get initial data
    initial_query_points = search_space.sample(num_initial_points)
    observer = mk_observer(function)
    initial_data = observer(initial_query_points)

    #Build model
    model = model_builder(initial_data, **model_args)

    #Bayesian Optimizer

    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    start = time.time()
    result = bo.optimize(
        num_steps,
        initial_data,
        model,
        acquisition_rule=acquisition_rule,
        track_state=False,
    )
    dataset = result.try_get_final_dataset()
    elapsed = time.time() - start

    #Get Results

    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()

    arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))
    found_minimum = observations[arg_min_idx, :]
    found_minimizer = query_points[arg_min_idx, :]

    results = {
        "objective": objective_name,
        "acquisition": acquisition_name,
        "num_initial_points": str(num_initial_points),
        "num_steps": str(num_steps),
        "seed": str(seed),
        "true_min": str(minimum.numpy()[0]),
        "true_minimizer": str(minimizer.numpy()),
        "found_minimum": str(found_minimum[0]),
        "found_minimizer": str(found_minimizer),
        "runtime": str(round(elapsed, 2))
    }
    for key, arg in model_args.items():
        results.update({key: str(arg)})

    #Write results
    os.makedirs(output_path, exist_ok = True)

    #Check for header of log_file
    header = ", ".join(results.keys()) + "\n"
    log_file = os.path.join(output_path, f"{model_name}_results.csv")

    try:
        with open(log_file, "r") as infile:
            file_header = infile.readlines(1)
            if file_header[0] != header:
                print(
                    f"Output log {log_file} already contains a header {file_header[0]}\n"
                    f"Which is not the same as the header for this simulation: \n"
                    f"{header} \n Continuing will overwrite the current log file entirely. "
                    f"Do you want to overwrite: y/n?"
                )
                overwrite = input()
                if overwrite.lower() == "y":
                    with open(log_file, "w") as output:
                        output.write(header)
                else: 
                    print("Exiting...")
                    exit()
                
    except FileNotFoundError:
        with open(log_file, "w") as output:
            output.write(header)
    
    # append results

    with open(log_file, "a") as output:
        output.write(", ".join(results.values()) + "\n")

    # Plot

    if plot:

        #Gen fig and save titles
        fig_title_suffix = [f"{key}: {model_args[key]}" for key in model_args.keys()]
        fig_title = f"{model_name} {acquisition_name} ({seed}) " + " ".join(fig_title_suffix)

        save_title_suffix = []
        for key in save_title_prefixes.keys():
            if key in model_args.keys():
                if isinstance(model_args[key], float):
                    arg = parse_rate(model_args[key])
                else:
                    arg = model_args[key]
                save_title_suffix.append(f"{save_title_prefixes[key]}{arg}")
        
        save_title = f"{model_name}_{acquisition_name}_s{seed}_" + "_".join(save_title_suffix)

        #Plot and save fitted plot
        fig = plot_function_plotly(
            function,
            search_space.lower,
            search_space.upper,
            grid_density=100,
            alpha=0.7,
        )
        fig.update_layout(height=800, width=800)

        fig = add_bo_points_plotly(
            x=query_points[:, 0],
            y=query_points[:, 1],
            z=observations[:, 0],
            num_init=num_initial_points,
            idx_best=arg_min_idx,
            fig=fig,
        )
        fig.update_layout(title = "fit " + fig_title)
        fig.write_html(os.path.join(output_path, f"{save_title}_fit.html"))
        print(f"{save_title}_fit saved!")

        #Plot and save predictions
        fig = plot_model_predictions_plotly(
            result.try_get_final_model(),
            search_space.lower,
            search_space.upper,
        )

        fig = add_bo_points_plotly(
            x=query_points[:, 0],
            y=query_points[:, 1],
            z=observations[:, 0],
            num_init=num_initial_points,
            idx_best=arg_min_idx,
            fig=fig,
            figrow=1,
            figcol=1,
        )
        fig.update_layout(height=800, width=800)
        fig.update_layout(title="predict " + fig_title)
        fig.write_html(os.path.join(output_path, f"{save_title}_predict.html"))
        print(f"{save_title}_predict saved!")


def multi_experiment(simul_args: dict):
    '''
    Run all cross possibilities of experiments.
    :param simul_args: A dict of all args
    '''
    for args in tqdm(itertools.product(*simul_args.values())):
        arg_dict = dict(zip(simul_args.keys(), args))
        simulate_experiment(**arg_dict)

