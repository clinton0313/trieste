#%%
import itertools
import numpy as np
import os
import pandas as pd
import pickle

from plotting_params import *
from typing import Callable, Tuple

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def nlpd(
    mean_pred: np.ndarray,
    var_pred: np.ndarray,
    query_points: np.ndarray, 
    objective: Callable
) -> np.ndarray:
    '''Computes Gaussian NLPD between predictions and query points'''
    true_points = objective(query_points)

    # epsilon = np.empty(var_pred.shape)
    # epsilon[:] = 1e-15
    # var_pred = np.max(np.stack((epsilon, var_pred), axis=1), axis=1)

    return (
        (mean_pred - true_points)**2 / (2 * var_pred)
        + 1/2 * np.log(2 * np.math.pi * var_pred)
    )


def gaussian_kernel_weights(
    minimizers: np.ndarray, 
    query_points: np.ndarray, 
    normalize: bool = True,
) -> np.ndarray:
    '''Computes Gaussina kernel weights to a minimum point form query points'''
    if len(minimizers.shape) != 2:
        minimizers = np.expand_dims(minimizers, axis=0)

    weights = np.min(
        np.stack(
            [
                np.exp(- np.linalg.norm(query_points - minimizers[i, :], axis=1))
                for i in range(minimizers.shape[0])
            ],
            axis=0
        ),
        axis = 0
    )
    
    if normalize:
        weights = weights / np.max(weights)

    return weights

def weighted_nlpd(
    record: pd.Series, 
    datadir: str, 
    obj_func_dict: dict = OBJ_FUNC_DICT,
    normalize: bool = True,
) -> Tuple[float, float, float, float]:
    '''For a record in a results dataframe, computes the starting weighted 
    nlpd and the end nlpd and returns as a tuple. '''
    
    with open(os.path.join(datadir, record["pickle_file"]), "rb") as infile:
        output = pickle.load(infile)

    predictions = output["predictions"]
    try:
        steps= list(predictions.keys())
    except KeyError:
        if int(predictions.keys()) == 0:
            steps = list(predictions.keys())
    total_nlpd = []
    for step in steps:
        neg_log_prob = nlpd(
            mean_pred = predictions[step]["mean"],
            var_pred = predictions[step]["var"],
            query_points = predictions[step]["coords"],
            objective = obj_func_dict[record["objective"]]
        )

        kernel_weights = gaussian_kernel_weights(
            minimizers = predictions[step]["minimizer"],
            query_points = predictions[step]["coords"],
            normalize = normalize
        )

        minimizer_nlpd = nlpd(
            mean_pred = predictions[step]["minimizer_mean"],
            var_pred = predictions[step]["minimizer_var"],
            query_points = predictions[step]["minimizer"],
            objective = obj_func_dict[record["objective"]]
        )

        minimizer_kernel_weights = gaussian_kernel_weights(
            minimizers = predictions[step]["minimizer"],
            query_points = predictions[step]["coords"] + 1e-15,
            normalize = normalize
        )
        

        total_nlpd.append(float(np.sum(neg_log_prob * kernel_weights)))
        total_nlpd.append(float(np.sum(minimizer_nlpd * minimizer_kernel_weights)))
    
    while len(total_nlpd) < 4:
        total_nlpd.append(np.nan)

    return tuple(total_nlpd)

def average_nlpd(
    results: pd.DataFrame,
    datadir: str,
    average_over: list = ["seed"],
    ignore_cols: list = [
        "true_min",
        "found_minimum",
        "num_initial_points",
        "steps_taken",
        "acquisition_runtime",
        "optimize_runtime",
        "pickle_file"
    ],
    **nlpd_kwargs,
) -> pd.DataFrame:
    """Computes nlpd statistics averaged over some columns (seed by default) of a 
    results dataframe. 

    :param results: pd.DataFrame of the CSV results
    :param datadir: Directory where the pickle files are stored
    :param average_over: Columns to average over, defaults to ["seed"]
    :param ignore_cols: Columns of dataframe to ignore when grouping by
    :param **nlpd_kwargs: Keyworkd args passed to weighted_nlpd function.

    :return: Grouped dataframe with regret statistics added. 
    """
    def get_nlpd(record: pd.Series) -> pd.Series:
        record["nlpd_start"], record["nlpd_end"] , record["nlpd_minimizer_start"], record["nlpd_minimizer_end"] = weighted_nlpd(record, datadir, **nlpd_kwargs)
        record["nlpd_difference"] = record["nlpd_end"] - record["nlpd_start"]
        record["nlpd_minimizer_difference"] = record["nlpd_minimizer_end"] - record["nlpd_minimizer_start"]
        return record
    
    groupby_cols = [col for col in results.columns if col not in (average_over + ignore_cols)]
    nlpd_cols = [
        "nlpd_start", 
        "nlpd_end",
        "nlpd_minimizer_start",
        "nlpd_minimizer_end",
        "nlpd_difference",
        "nlpd_minimizer_difference"
    ]
    all_nlpd_cols = [
        f"{colname}_{stat}"
        for colname, stat  in itertools.product(nlpd_cols, ["mean", "var"])
    ]

    grouped_results = pd.DataFrame(columns=groupby_cols + all_nlpd_cols)
    results = results.drop_duplicates()
    for groupby_values in itertools.product(
        *map(lambda x: results[x].unique(), groupby_cols)
    ):
        #Bunch of munging to filter dataframe to rows meeting the groupby conditions
        groupby_value_lists = map(lambda x: [x], groupby_values)
        groupby_dict = dict(zip(groupby_cols, groupby_value_lists))
        groupby_mask = results.isin(groupby_dict)[groupby_cols]
        filtered_results = results[groupby_mask.all(1)]

        filtered_results = filtered_results.apply(get_nlpd, axis=1)
        agg_results = dict(zip(groupby_cols, groupby_values))
        for i, col in enumerate(nlpd_cols):
            try:
                stats = filtered_results.loc[:, col]
            except KeyError:
                continue
            
            mean_stat = np.nanmean(stats)
            agg_results.update({all_nlpd_cols[2 * i]: mean_stat})
            var_stat = np.nanvar(stats)
            agg_results.update({all_nlpd_cols[2 * i + 1]: var_stat})
            
        grouped_results = grouped_results.append(agg_results, ignore_index=True)
    
    return grouped_results

def get_all_nlpd(
    results_dir: str,
    models_dict: dict,
    **nlpd_kwargs
)-> pd.DataFrame:

    all_results = []
    for model_dir, model_meta in models_dict.items():
        if model_dir == "random":
            continue
        data_dir = os.path.join(results_dir, model_dir)
        results = pd.read_csv(os.path.join(data_dir, f"{model_meta['name']}_results.csv"), skipinitialspace=True)
        grouped_results = average_nlpd(results, data_dir, **nlpd_kwargs)
        all_results.append(grouped_results)
    
    return pd.concat(all_results, ignore_index=True)
# %%

if __name__ == "__main__":
    # model_name = "gpr"
    # DATADIR = os.path.join(RESULTS_DIR, model_name)
    # results = pd.read_csv(os.path.join(DATADIR, f"{model_name}_results.csv"), skipinitialspace=True)
    # grouped_results = average_nlpd(results, DATADIR)
    # print(grouped_results)
    all_results = get_all_nlpd(RESULTS_DIR, MODELS_DICT)
    all_results.to_csv(os.path.join(RESULTS_DIR, "nlpd_results.csv"))
    print(all_results)
# %%
