# %%
from benchmarking.benchmarking_utils import *
import pandas as pd
import numpy as np
import os, json
from glob import glob
# %%
PATH = "results"
COLUMNS = [
    "model", 
    "objective", 
    "acquisition", 
    "seed", 
    "true_min", 
    "found_minimum", 
    "steps_taken",  
    "optimize_runtime",
    "acquisition_runtime"
]

csvs = [
    file for path, _, _ in os.walk(PATH) for file in glob(os.path.join(path, "*.csv")) 
    if "noisy" not in file 
    if "nlpd" not in file
    if "random" not in file
]
with open('output_ranges.json') as f:
    output_ranges = json.load(f)

# %%
def produce_stats(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file, usecols=COLUMNS)
        df = (
            df.assign(
                optimize_runtime_step = df["optimize_runtime"] / (df["steps_taken"] + 1),
                acquisition_runtime_step = df["acquisition_runtime"] / (df["steps_taken"] + 1),
                final_regret = np.abs(df["found_minimum"] - df["true_min"]) / df["objective"].map(output_ranges),
                found_min = lambda dtf: np.where(dtf.final_regret < 1e-03, 1, 0)
            )
        )
        dfs.append(df)

    df = pd.concat(dfs)
    stats = (
        df[["model", "objective", "acquisition", "acquisition_runtime_step", "optimize_runtime_step", "steps_taken", "final_regret", "found_min"]]
        # .loc[lambda df: ~df.objective.str.contains("noisy")]
    )

    stats = (
        stats
        .groupby(["objective", "model", "acquisition"])
        .describe()
        .loc[:, lambda df: df.columns.get_level_values(1).isin({"mean", "std"})]
    )

    stats.loc[:, lambda df: df.columns.get_level_values(1)=="mean"] = (
        stats.loc[:, lambda df: df.columns.get_level_values(1)=="mean"].round(2)
    )

    stats.loc[:, lambda df: df.columns.get_level_values(1)=="std"] = (
        stats.loc[:, lambda df: df.columns.get_level_values(1)=="std"].round(2)
    )

    stats = stats.drop(columns=[("found_min", "std"), ("final_regret", "std")])

    return stats, df
# %%
stats, df = produce_stats(csvs)
stats.to_csv("stats.csv")



# %% Plot results
import matplotlib.pyplot as plt
import seaborn as sns

df = df.loc[df["seed"]<=10]
df.reset_index(inplace=True, drop=True)
df.drop_duplicates(subset=["model", "objective", "seed", "acquisition"], inplace=True)
palette ={"de": "tab:green", "new_der_log": "tab:orange", "deup": "tab:blue", "gpr": "tab:red", "mc": "tab:gray"}


def plot_regret(df, acq):
    fig, ax = plt.subplots(
        5,2, 
        figsize=(12,16),
        sharex=False, 
        sharey=True, 
        gridspec_kw={'hspace': 0.4, 'wspace': 0.15}
    )


    for axis, objective in zip(ax.reshape(-1), df.objective.unique()):
        tmp_ = df.loc[(df["objective"]==objective) & (df["acquisition"]==acq)]

        sns.barplot(
            data=tmp_, 
            x="model", 
            y="final_regret", 
            hue="seed", 
            ax=axis, 
            palette="gray", 
            ci=None,
            linewidth=1,
            edgecolor=".2"
        )

        axis.get_legend().remove()
        axis.set_title(objective)
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    fig.suptitle("Final Regret with acquisition:" + acq, fontsize=24)
    return fig



def plot_steps(df, acq):
    fig, ax = plt.subplots(
        5,2, 
        figsize=(12,16),
        sharex=False, 
        sharey=True, 
        gridspec_kw={'hspace': 0.4, 'wspace': 0.15}
    )


    for axis, objective in zip(ax.reshape(-1), df.objective.unique()):
        tmp_ = df.loc[(df["objective"]==objective) & (df["acquisition"]==acq)]

        sns.lineplot(
            data=tmp_, 
            x="seed",
            y="steps_taken",
            hue="model",
            palette=palette,
            ax=axis,
            markers=True
        )

        axis.get_legend().remove()
        axis.set_title(objective)
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    handles, labels = ax[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=16)
    fig.suptitle("Steps taken with acquisition:" + acq, fontsize=24)

    return fig


def plot_optim_runtime(df):
    fig, ax = plt.subplots(
        5,2, 
        figsize=(12,16),
        sharex=False, 
        sharey=True, 
        gridspec_kw={'hspace': 0.4, 'wspace': 0.15}
    )


    for axis, objective in zip(ax.reshape(-1), df.objective.unique()):
        tmp_ = df.loc[df["objective"]==objective]

        sns.barplot(
            data=tmp_,
            x="model",
            y=("optimize_runtime_step", "mean"),
            ax=axis
        )

        axis.set_title(objective)
        axis.set_ylabel("Optimize runtime per step")
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    fig.suptitle("Optimize runtime per step", fontsize=24)

    return fig


# %%
fig = plot_regret(df, "ts")
fig.savefig('results/final_regret_stats_ts.png', bbox_inches='tight', dpi=300)

fig = plot_regret(df, "ei")
fig.savefig('results/final_regret_stats_ei.png', bbox_inches='tight', dpi=300)

# %%
fig = plot_steps(df, "ts")
fig.savefig('results/steps_taken_stats_ts.png', bbox_inches='tight', dpi=300)

fig = plot_steps(df, "ei")
fig.savefig('results/steps_taken_stats_ei.png', bbox_inches='tight', dpi=300)
# %%
stats.reset_index(inplace=True)
fig = plot_optim_runtime(stats)
fig.savefig('results/optim_runtime.png', bbox_inches='tight', dpi=300)
# %%
