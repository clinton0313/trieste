from collections import defaultdict
import gpflow
import itertools
import numpy as np
import os
import pickle
import platform
import psutil
import random
import tensorflow as tf
import tensorflow_probability as tfp 
import timeit
import trieste

from tqdm import tqdm
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.models.keras import (
    DirectEpistemicUncertaintyPredictor,
    DeepEvidentialRegression,
    MonteCarloDropout,
    DeepEnsemble,
    build_vanilla_deup,
    build_vanilla_keras_ensemble,
    build_vanilla_keras_mcdropout,
    build_vanilla_keras_deep_evidential, 
)
from trieste.models.keras.architectures import DropConnectNetwork, EpistemicUncertaintyNetwork
from trieste.objectives import (
    michalewicz_2,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_2_SEARCH_SPACE,
    MICHALEWICZ_2_MINIMIZER,
    scaled_branin,
    SCALED_BRANIN_MINIMUM,
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE,
    dropwave,
    DROPWAVE_MINIMUM,
    DROPWAVE_MINIMIZER,
    DROPWAVE_SEARCH_SPACE,
    eggholder,
    EGGHOLDER_MINIMUM,
    EGGHOLDER_MINIMIZER,
    EGGHOLDER_SEARCH_SPACE
)
from trieste.objectives.single_objectives import HARTMANN_6_SEARCH_SPACE, HARTMANN_6_MINIMIZER, HARTMANN_6_MINIMUM, hartmann_6
from trieste.objectives.utils import mk_observer
from trieste.models.optimizer import KerasOptimizer
from trieste.ask_tell_optimization import AskTellOptimizer
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
    "e_num_hidden_layers": "eL",
    "e_units": "en",
    "reg_weight": "rw",
    "maxi_rate": "m",
    "rate": "dr",
    "lr": "lr",
    "ensemble_size": "es"
}

# OBJECTIVES

michal2 = ("michal2", michalewicz_2, MICHALEWICZ_2_SEARCH_SPACE, MICHALEWICZ_2_MINIMUM, MICHALEWICZ_2_MINIMIZER)
dropw2 = ("dropw2", dropwave, DROPWAVE_SEARCH_SPACE, DROPWAVE_MINIMUM, DROPWAVE_MINIMIZER)
eggho2 = ("eggho2", michalewicz_2, EGGHOLDER_SEARCH_SPACE, EGGHOLDER_MINIMUM, EGGHOLDER_MINIMIZER)
branin = ("scaled_branin", scaled_branin, BRANIN_SEARCH_SPACE, SCALED_BRANIN_MINIMUM, BRANIN_MINIMIZERS)
hartmann6 = ("hartmann6", hartmann_6, HARTMANN_6_SEARCH_SPACE, HARTMANN_6_MINIMUM, HARTMANN_6_MINIMIZER)

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


def deup_builder(data, ensemble_size, num_hidden_layers, units, lr, e_num_hidden_layers=4, e_units=128):
    f_keras_ensemble, e_predictor = build_vanilla_deup(
        data, 
        f_model_builder=build_vanilla_keras_ensemble,
        ensemble_size=ensemble_size,
        num_hidden_layers=num_hidden_layers,
        units=units,
        activation="relu",
        independent_normal=False,
        e_num_hidden_layers=e_num_hidden_layers,
        e_units=e_units,
        e_activation="relu"
    )

    f_ensemble = DeepEnsemble(f_keras_ensemble, optimizer=KerasOptimizer(tf.optimizers.Adam(lr)),)

    deup = DirectEpistemicUncertaintyPredictor(
        model={"f_model": f_ensemble, "e_model": e_predictor},
        optimizer=KerasOptimizer(tf.optimizers.Adam(lr)), init_buffer=True
    )

    return deup


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
    if rate < 10 and rate > -10:
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


def check_csv_header(header, log_file):
    '''Checks for consistency of CSV header'''
    try:
        with open(log_file, "r") as infile:
            file_header = infile.readlines(1)
            if file_header[0] != header:
                tqdm.write(
                    f"Output log {log_file} already contains a header {file_header[0]}\n"
                    f"Which is not the same as the header for this simulation: \n"
                    f"{header} \n Continuing will overwrite the current log file entirely. "
                    f"Do you want to overwrite: y/n?"
                )
                overwrite_csv = input()
                if overwrite_csv.lower() == "y":
                    with open(log_file, "w") as outfile:
                        outfile.write(header)
                else: 
                    tqdm.write("Exiting...")
                    exit()
                
    except FileNotFoundError:
        with open(log_file, "w") as outfile:
            outfile.write(header)


def default_metadata() -> str:
    return (

        f"""
        Computer network name: {platform.node()}
        Machine type: {platform.machine()}
        Processor type: {platform.processor()}
        Platform type: {platform.platform()}
        Number of physical cores: {psutil.cpu_count(logical=False)}
        Number of logical cores: {psutil.cpu_count(logical=True)}
        Total RAM installed: {round(psutil.virtual_memory().total/1000000000, 2)} GB
        Available RAM: {round(psutil.virtual_memory().available/1000000000, 2)} GB\n
        """
    )

def save_plotly(
    num_initial_points, 
    output_path, 
    save_title,
    seed, 
    model_args, 
    acquisition_name, 
    model_name, 
    function, 
    search_space, 
    result, 
    query_points, 
    observations, 
    arg_min_idx
):
    """Plots and saves the fitted and predicted plotly figures to output path"""
    fig_title_suffix = [f"{key}: {model_args[key]}" for key in model_args.keys()]
    fig_title = f"{model_name} {acquisition_name} ({seed}) " + " ".join(fig_title_suffix)

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
    tqdm.write(f"{save_title}_fit saved!")

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
    tqdm.write(f"{save_title}_predict saved!")

def make_predictions(model, search_space, grid_density=20) -> Tuple[list, list, list , list]:

    '''
    Makes predictions over a grid of the search_space. Ready to be plotted with plot_predictions_plotly. 
    :returns: Xplot, mean, var
    '''

    mins = search_space.lower.numpy()
    maxs = search_space.upper.numpy()

    coord_ranges = [np.linspace(mins[i], maxs[i], grid_density) for i in range(mins.shape[0])]

    mesh_coords = np.meshgrid(*coord_ranges)
    Xplot = np.vstack([coord.flatten() for coord in mesh_coords]).T

    Fmean, Fvar = model.predict(Xplot)

    return Xplot, Fmean, Fvar

def simulate_experiment(
    objective: Tuple, 
    num_initial_points: int,
    acquisition: Tuple[str, trieste.acquisition.rule.AcquisitionRule],
    num_steps: int,
    predict_interval: int,
    model: Tuple[str, Callable],
    output_path: str,
    metadata: str,
    grid_density: int = 20,
    save_title_prefixes: dict = global_save_title_prefixes,
    plot: bool = False,
    report_predictions: bool = True,
    overwrite: bool = False,
    seed: int = 0,
    **model_args
):
    """
    :param objective: Tuple of (objective_name, function, search_space, minimum, minimizer)
    :param num_initial_points: Number of initial query points.
    :param acquisition: Tuple of (acquisition_name, instantiated Acquisition rule)
    :param num_steps: Number of bayesian optimization steps.
    :param predict_interval: Interval between number of BO steps to predict the entire surface. 
    :param model: Tuple of (model_name, model_builder). Model_builder is a function that accepts
        arguments: initial_data, and **model_args and returns a model. 
    :param output_path: Path to save figs and log_file
    :param metadata: A string to be saved in separate metadata txt file. 
    :param grid_density: Density of grid for predictions. 
    :param save_title_prefixes: Dictionary of {model_arg: prefix} used for prefixing the save_title. For
        example: {'num_hidden_layers': 'hl'}. Will filter out unused prefixes hence defaults to 
        global_save_title_prefixes where we can add prefixes for all models. 
    :param plot: True or False, whether to generate plots or not. 
    :param report_predictions: True or False, whether to generate predictions or not. Defaults to True. 
    :param overwrite: If True overwrites prediction pickles that match. Defaults to False.
    :param seed: Seed.
    """
    #Set seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    #Unpack model, acquisition and objective tuples:
    acquisition_name, acquisition_rule = acquisition
    model_name, model_builder = model
    objective_name, function, search_space, minimum, minimizer = objective
    #Make output path
    os.makedirs(output_path, exist_ok = True)

    #Get save_title
    save_title_suffix = []
    for key in save_title_prefixes.keys():
        if key in model_args.keys():
            if isinstance(model_args[key], float):
                arg = parse_rate(model_args[key])
            else:
                arg = model_args[key]
            save_title_suffix.append(f"{save_title_prefixes[key]}{arg}")
        
    save_title = f"{model_name}_{acquisition_name}_{objective_name}_" + "_".join(save_title_suffix) + f"_seed{seed}"
    
    #Check if this iteration already done and skip if pkl already exists. 
    if not overwrite:
        if os.path.isfile(os.path.join(output_path, f"{save_title}.pkl")):
            tqdm.write(f"Pickle file found for {save_title}.pkl! Skipping this simulation...")
            return

    #Get initial data
    initial_query_points = search_space.sample_sobol(num_initial_points)
    observer = mk_observer(function)
    initial_data = observer(initial_query_points)

    #Build model
    built_model = model_builder(initial_data, **model_args)

    #Bayesian Optimizer
    ask_tell = AskTellOptimizer(search_space, initial_data, built_model, acquisition_rule=acquisition_rule)

    predictions = defaultdict(dict)
    optimize_time = 0
    acquisition_time = 0
    for step in range(num_steps):
        #Basic loop
        start_acquisition_time = timeit.default_timer()
        new_point = ask_tell.ask()
        acquisition_time += start_acquisition_time - timeit.default_timer()

        new_data = observer(new_point)

        start_optimize_time = timeit.default_timer()
        ask_tell.tell(new_data)
        optimize_time += timeit.default_timer() - start_optimize_time

        #Predictions
        if report_predictions:
            if step % predict_interval == 0 or step == num_steps - 1:
                current_model = ask_tell.to_result(copy=False).try_get_final_model()
                prediction = make_predictions(current_model, search_space, grid_density=grid_density)
                predictions[step]["coords"], predictions[step]["mean"], predictions[step]["var"] = prediction


    result = ask_tell.to_result(copy=False)
    dataset = result.try_get_final_dataset()

    #Get Results
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()

    arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))
    found_minimum = observations[arg_min_idx, :]

    results = {
        "objective": objective_name,
        "acquisition": acquisition_name,
        "num_initial_points": str(num_initial_points),
        "num_steps": str(num_steps),
        "seed": str(seed),
        "true_min": str(minimum.numpy()[0]),
        "found_minimum": str(found_minimum[0]),
        "acquisition_runtime": str(round(acquisition_time, 3)),
        "optimize_runtime": str(round(optimize_time, 3)),
    }
    for key, arg in model_args.items():
        results.update({key: str(arg)})
    results.update({"pickle_file": f"{save_title}.pkl"})
    
    #Write results

    #Check for header of log_file
    header = ",".join(results.keys()) + "\n"
    log_file = os.path.join(output_path, f"{model_name}_results.csv")
    check_csv_header(header, log_file)
    
    # append results
    with open(log_file, "a") as outfile:
        outfile.write(",".join(results.values()) + "\n")
    tqdm.write(f"{results.values()} appended to {log_file}!")

    #Save outputdata
    if report_predictions:
        output_data = {
            "query_points": query_points,
            "observations": observations,
            "predictions": predictions,
            "metadata": default_metadata() + metadata
        }
        with open(os.path.join(output_path, f"{save_title}.pkl"), "wb") as outfile:
            pickle.dump(output_data, outfile)
        tqdm.write(f"Output data saved for {save_title} model. ")

    # Plot
    if plot:

        #Gen fig and save titles
        save_plotly(
            num_initial_points, 
            output_path, 
            save_title, 
            seed, 
            model_args, 
            acquisition_name, 
            model_name, 
            function, 
            search_space, 
            result, 
            query_points, 
            observations, 
            arg_min_idx
        )


def multi_experiment(
    simul_args: dict,  
    disable_tqdm: bool=False
):
    '''
    Run all cross possibilities of experiments.
    :param simul_args: A dict of all args for simulate_expriemnt
    :param disable_tqdm: self-explanatory

    simulate experiment args:

        :param objective: Tuple of (objective_name, function, search_space, minimum, minimizer)
        :param num_initial_points: Number of initial query points.
        :param acquisition: Tuple of (acquisition_name, instantiated Acquisition rule)
        :param num_steps: Number of bayesian optimization steps.
        :param predict_interval: Interval between number of BO steps to predict the entire surface. 
        :param model: Tuple of (model_name, model_builder). Model_builder is a function that accepts
            arguments: initial_data, and **model_args and returns a model. 
        :param output_path: Path to save figs and log_file
        :param metadata: A string to be saved in separate metadata txt file. 
        :param grid_density: Density of grid for predictions. 
        :param save_title_prefixes: Dictionary of {model_arg: prefix} used for prefixing the save_title. For
            example: {'num_hidden_layers': 'hl'}. Will filter out unused prefixes hence defaults to 
            global_save_title_prefixes where we can add prefixes for all models. 
        :param plot: True or False, whether to generate plots or not. 
        :param predictions: True or False, whether to generate predictions or not. Defaults to True. 
        :param overwrite: If True overwrites prediction pickles that match. Defaults to False.
        :param seed: Seed.
    '''
    for key, arg in simul_args.items():
        if not isinstance(arg, list):
            simul_args.update({key: [arg]})

    for args in tqdm(
        itertools.product(*simul_args.values()),
        colour="blue",
        disable=disable_tqdm
    ):
        arg_dict = dict(zip(simul_args.keys(), args))
        simulate_experiment(**arg_dict)

