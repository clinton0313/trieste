#%%
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from typing import Tuple, Callable

pd.set_option("display.max_rows", 300)
matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
})
matplotlib.style.use("seaborn-bright")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

model_name = "random"
DATADIR = os.path.join("test_parallel_benchmarking", model_name)

results = pd.read_csv(os.path.join(DATADIR, f"{model_name}_results.csv"), skipinitialspace=True)
# %%
def simple_regret(record: pd.Series, datadir: str = DATADIR) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the simple regret for a row of the results csv by opening
    the pickle file and parsing. 

    :param record: A row of the results csv.
    :param datadir: Directory where the pickle files are stored. 
    :return: A tuple of bayesian optimization steps and simple regret
    """

    with open(os.path.join(datadir, record["pickle_file"]), "rb") as infile:
        output = pickle.load(infile)
    
    steps = np.arange(len(output["observations"]))
    regret = np.abs(output["observations"] - record["true_min"])

    return steps, regret.squeeze()

def minimum_regret(record: pd.Series, datadir: str = DATADIR) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the cumulative regret for a row of the results csv by opening
    the pickle file and parsing. 

    :param record: A row of the results csv.
    :param datadir: Directory where the pickle files are stored. 
    :return: A tuple of bayesian optimization steps and cumulative regret
    """

    steps, regret = simple_regret(record, datadir)
    cumulative_regret = np.minimum.accumulate(regret)
    
    return steps, cumulative_regret

def expand_regret(
    record: pd.Series, 
    regret_function: Callable, 
    max_steps: int, 
    datadir:str = DATADIR
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad regret with nan's to match the length of max_steps.

    :param record: pd.Series of a row of a results csv
    :param regret_function: Regret function that returns steps and regret
    :param max_steps: Max number of BO steps to match sequence to
    :param datadir: Directory where the pickle files are stored, defaults to DATADIR
    :return: Tuple of padded steps and regret
    """
    steps, regret = regret_function(record, datadir)
    try:
        expansion = np.zeros(max_steps - len(steps))
        expansion[:] = np.nan
        return (
            np.concatenate((steps, expansion)),
            np.concatenate((regret, expansion))
        )
    except ValueError:
        return(
            steps[:max_steps],
            regret[:max_steps]
        )


def average_regret(
    results: pd.DataFrame, 
    average_over: list = ["seed"], 
    ignore_cols: list = [
        "true_min", 
        "found_minimum", 
        "steps_taken", 
        "acquisition_runtime", 
        "optimize_runtime", 
        "pickle_file"
    ],
    regret_function: Callable = minimum_regret,
    regret_name: str = "min_regret",
    compute_var: bool = True, 
    max_steps: int = 0,
    datadir: str = DATADIR
) -> pd.DataFrame:
    """Computes regret statistics averaged over some columns (seed by default). 

    :param results: pd.DataFrame of the CSV results
    :param average_over: Columns to average over, defaults to ["seed"]
    :param ignore_cols: Columns of dataframe to ignore when grouping by
    :param regret_function: Regret function that returns steps and regret,
        defaults to minimum_regret
    :param regret_name: Naming used for regret function, defaults to "min_regret"
    :param compute_var: Whether or not to compute the variance, defaults to True
    :param max_steps: Maximum number of steps to match regret to,
         defaults to 0 which will infer the maximum number of steps + initial points
         from the dataframe. 
    :param datadir: Directory where the pickle files are stored, defaults to DATADIR
    :return: Grouped dataframe with regret statistics added. 
    """
    if max_steps == 0:
        max_steps = results.num_steps.max() + results.num_initial_points.max()

    def get_regret(record: pd.Series):
        record["steps"], record["regret"] = expand_regret(record, regret_function, max_steps, datadir)
        return record

    groupby_cols = [col for col in results.columns if col not in (average_over + ignore_cols)]
    regret_cols = [f"mean_{regret_name}"]
    if compute_var:
        regret_cols.append(f"var_{regret_name}")


    grouped_results = pd.DataFrame(columns=groupby_cols + regret_cols)

    for groupby_values in itertools.product(
        *map(lambda x: results[x].unique(), groupby_cols)
    ):
        #Bunch of munging to filter dataframe to rows meeting the groupby conditions
        groupby_value_lists = map(lambda x: [x], groupby_values)
        groupby_dict = dict(zip(groupby_cols, groupby_value_lists))
        groupby_mask = results.isin(groupby_dict)[groupby_cols]
        filtered_results = results[groupby_mask.all(1)]

        filtered_results = filtered_results.apply(get_regret, axis=1)
        agg_results = dict(zip(groupby_cols, groupby_values))
        try:
            regrets = np.stack(filtered_results.loc[:, "regret"], axis=1)
        except KeyError:
            continue
        
        mean_regret = np.nanmean(regrets, axis=1)
        agg_results.update({regret_cols[0]: mean_regret})
        if compute_var: 
            var_regret = np.nanvar(regrets, axis=1)
            agg_results.update({regret_cols[1]: var_regret})
        
        grouped_results = grouped_results.append(agg_results, ignore_index=True)
    
    return grouped_results

# %%

def plot_regret(
    mean: np.ndarray,
    var: np.ndarray = None,
    n_stds: int = 3,
    ax = None,
    title: str = "",
    label: str = "",
    xlabel:str = "Steps",
    ylabel: str = "Regret",
    xlim: tuple = None,
    ylim: tuple = None,
    regret_color: str = "blue",
    std_alpha: float = 0.3,
    **plot_kwargs
):
    matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
    })
    matplotlib.style.use("seaborn-bright")

    if ax is None:
        fig, ax = plt.subplots(**plot_kwargs)
    
    steps = np.arange(len(mean))
    ax.plot(steps, mean, color=regret_color, label=label)
    if var is not None:
        for n in range(n_stds):

            ax.fill_between(
                steps, 
                mean - (n+1) * var**0.5, 
                mean + (n+1) * var**0.5, 
                color=regret_color, 
                alpha=std_alpha
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0,)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0,)

    if label != "":
        ax.legend()

    try:
        return fig
    except UnboundLocalError:
        pass

#%%

if __name__ == "__main__":
    grouped_results = average_regret(results)
    fig = plot_regret(grouped_results.loc[0, "mean_min_regret"], grouped_results.loc[0, "var_min_regret"])
    plt.show()

    f, a = plt.subplots()

    for i in range(5):
        plot_regret(
            grouped_results.loc[i, "mean_min_regret"],
            grouped_results.loc[i, "var_min_regret"],
            regret_color = list(matplotlib.colors.TABLEAU_COLORS.keys())[i],
            label= grouped_results.loc[i, "objective"],
            ax=a
        )
    plt.show()
# %%
