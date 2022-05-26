# %%

import pickle, os
from pathlib import Path
from benchmarking.results_processing.utils.plotting_params import COLOR_DICT, MODELS_DICT, OBJECTIVE_DICT
import numpy as np
import pandas as pd
from itertools import product
import matplotlib
import matplotlib.pyplot as plt

from utils.plotting_params import *

basedir = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
# %%
results = []
functions = ["noisy_shekel4", "noisy_hartmann6"] # "noisy_ackley5", 
models = ["de", "der", "mc", "deup", "svgp", "gpr"]

def get_optimtime(record, name):
    steps = np.arange(len(record["optimize_timelist"]))*100
    return pd.Series(record["optimize_timelist"], index=steps, name=name)

for function, model in product(functions, models):
    files = [x for x in os.listdir(basedir / "results" / model) if function in x]

    ls = []
    for idx, file in enumerate(files):
        with open(os.path.join(basedir, "results", model, file), "rb") as infile:
            output = pickle.load(infile)
        ls.append(get_optimtime(output, idx))

    df = pd.DataFrame(ls).T
    mean_ = pd.Series(df.mean(axis=1), name=function + "_" + model + "_mean")
    var_ = pd.Series(df.std(axis=1), name=function + "_" + model + "_std")
    results.append(mean_); results.append(var_) 

results = pd.DataFrame(results).T

# # Stretch
# gpr = results.loc[:, results.columns.str.contains("_gpr_")]
# gpr.index = gpr.index/10
# gpr.index = gpr.index.astype(int)

# results = results.loc[:, ~(results.columns.str.contains("_gpr_"))]
stretch_results = pd.DataFrame(index=np.arange(0, 4910, 10))
stretch_results = stretch_results.merge(results, left_index=True, right_index=True, how="left")
results = stretch_results.interpolate(method="linear")
# results = stretch_results.merge(gpr, left_index=True, right_index=True, how="left")
# %%

def plot_optimtime(
    mean: pd.Series,
    std: pd.Series = None,
    n_stds: int = 1,
    std_alpha: float = 0.3,
    ax = None,
    title: str = "",
    label: str = "",
    xlabel:str = "Steps",
    ylabel: str = "Optimization Step Runtime (seconds)",
    runtime_color: str = "blue",
    xlim: tuple = None,
    ylim: tuple = None,
    log_scale: bool = False,
    idx_pos = None,
    **kwargs
):
    matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
    })
    matplotlib.style.use("seaborn-bright")

    if ax is None:
        fig, ax = plt.subplots()
    
    steps = np.arange(mean.shape[0])
    if "gpr" in mean.columns:
        mean.index = steps
    ax.plot(mean.index, mean.values, color=runtime_color, label=label, **kwargs)
    if std is not None:
        for n in range(n_stds):

            ax.fill_between(
                mean.index, 
                np.squeeze(mean.values - (n+1) * std.values), 
                np.squeeze(mean.values + (n+1) * std.values), 
                color=runtime_color, 
                alpha=std_alpha
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if idx_pos == 0:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0,5000)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0,1500)

    # if log_scale:
    #     ax.set_yscale("log")

    if label != "" and idx_pos == 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.3,1))

    try:
        return fig
    except UnboundLocalError:
        pass
# %%
# plot_optimtime(results["hartmann6_de_mean"], results["hartmann6_de_std"])
# %%

def plot_optimtime_comparison(
    data: pd.DataFrame,
    objective: str,
    ax,
    model_dict = MODELS_DICT,
    objective_label_dict = OBJECTIVE_DICT,
    color_dict = COLOR_DICT,
    **kwargs
):
    data = data.loc[:, data.columns.str.contains(objective + "_")]
    for model_label, model_meta in model_dict.items():
        if model_label != "random":
            data_ = data.loc[:, data.columns.str.contains("_" + model_label + "_")]
            mean = data_.loc[:, data_.columns.str.contains("_mean")]
            std = data_.loc[:, data_.columns.str.contains("_std")]

            plot_optimtime(
                mean,
                std,
                ax=ax,
                title=f"{objective_label_dict[objective]}",
                label=model_meta['label'],
                runtime_color = color_dict[model_label],
                **kwargs
            )

    

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    SAVEPATH = os.path.join("figs", "optimtime_plots")
    f, a = plt.subplots(1,2, figsize=(12,6),sharey=True)
    for id, function in enumerate(functions):
        plot_optimtime_comparison(
            results,
            function,
            ax=a[id],
            idx_pos=id,
            log_scale=False
        )

    # # Include cubic projection
    # res = results[function + "_gpr_mean"].values[~np.isnan(results[function + "_gpr_mean"].values)]
    # poly = np.polyfit(np.arange(0,500,10), res, deg=3)
    # projection = np.polyval(poly, np.arange(500,1000,10))

    # a2 = f.add_subplot(111, label="projection", frame_on=False)
    # a2.plot(np.arange(500,1000,10), projection, color="r", linestyle="dashed", alpha=.5)
    # a2.set_xticks([])
    # a2.set_yticks([])
    # a2.set_ylim(0,1500)
    # a2.set_xlim(0,5000)

    f.savefig(
        os.path.join(SAVEPATH, "model_optimization_comparison.png"),
        facecolor="white",
        transparent=False,
        bbox_inches='tight'
    )
    f.clear()
    plt.close(f)
# %%
