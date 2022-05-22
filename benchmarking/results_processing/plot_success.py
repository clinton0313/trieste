#%%
from typing import Sequence
import matplotlib
from matplotlib import tight_layout
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
})
matplotlib.style.use("seaborn-bright")
matplotlib.use("tkagg")
os.chdir(os.path.dirname(os.path.realpath(__file__)))



stats = pd.read_csv("stats_flattened.csv")
SAVEPATH = "figs"

#%%

def quick_bar_plots(
    x: str,
    y: str,
    hue: str,
    title: str,
    stats: pd.DataFrame,
    savepath: str = SAVEPATH,
    plot_by: str = "model",
    fig_rows: int = 3,
    fig_cols: int = 2,
    log_scale: bool = False
) -> None:
    """
    Plots bar plots for every model and an overall barplot, as well as gridded.

    :param x: x column name in dataframe
    :param y: y column name in dataframe
    :param hue: hue column name in dataframe
    :param title: titling for save directory and filenames
    :param stats: Dataframe of results
    :param savepath: Savepath, defaults to SAVEPATH (figs)
    """
    os.makedirs(os.path.join(savepath, title), exist_ok=True)
    for model in stats[plot_by].unique():
        f, a = plt.subplots()
        sns.barplot(x, y, hue=hue, data=stats[stats[plot_by]==model], ax=a)
        a.set_title(f"{model}")
        f.savefig(os.path.join(savepath, title, f"{model}_{title}.png"), facecolor="white", transparent=False)
        f.clear()
        plt.close(f)

    f, a = plt.subplots()
    sns.barplot(x, y, hue=hue, data=stats, ax=a)
    a.set_title("Overall")
    if log_scale:
        a.set_yscale("log")
    f.savefig(os.path.join(savepath, title, f"overall_{title}.png"), facecolor="white", transparent=False)

    f, ax = plt.subplots(fig_rows, fig_cols, tight_layout=True)
    sns.barplot(x, y, hue=hue, data=stats, ax=ax.ravel()[0])
    ax.ravel()[0].set_xticklabels(ax.ravel()[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.ravel()[0].set_title("Overall")
    if log_scale:
        ax.ravel()[0].set_yscale("log")
    for model, a in zip(stats[plot_by].unique(), ax.ravel()[1:]):
        sns.barplot(x, y, hue=hue, data=stats[stats[plot_by]==model], ax=a)
        a.set_title(f"{model}")
        a.set_xticklabels(a.get_xticklabels(), rotation=45, horizontalalignment='right')
        if log_scale:
            a.set_yscale("log")
    f.savefig(os.path.join(savepath, title, f"all_{title}.png"), facecolor="white", transparent=False)


if __name__ == "__main__":
    quick_bar_plots("steps_taken_mean", "objective", "acquisition", "steps", stats)