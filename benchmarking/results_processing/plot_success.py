#%%
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

#%%

def quick_bar_plots(x, y, hue, title, stats=stats):
    os.makedirs(os.path.join("figs", title), exist_ok=True)
    for model in stats["model"].unique():
        f, a = plt.subplots()
        sns.barplot(x, y, hue=hue, data=stats[stats["model"]==model], ax=a)
        a.set_title(f"{model}")
        f.savefig(os.path.join("figs", title, f"{model}_{title}.png"), facecolor="white", transparent=False)
        f.clear()
        plt.close(f)

    f, a = plt.subplots()
    sns.barplot(x, y, hue=hue, data=stats, ax=a)
    a.set_title("Overall")
    f.savefig(os.path.join("figs", title, f"overall_{title}.png"), facecolor="white", transparent=False)

    f, ax = plt.subplots(3, 2, tight_layout=True)
    sns.barplot(x, y, hue=hue, data=stats, ax=ax.ravel()[0])
    ax.ravel()[0].set_xticklabels(ax.ravel()[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.ravel()[0].set_title("Overall")
    for model, a in zip(stats["model"].unique(), ax.ravel()[1:]):
        sns.barplot(x, y, hue=hue, data=stats[stats["model"]==model], ax=a)
        a.set_title(f"{model}")
        a.set_xticklabels(a.get_xticklabels(), rotation=45, horizontalalignment='right')
    f.savefig(os.path.join("figs", title, f"all_{title}.png"), facecolor="white", transparent=False)

quick_bar_plots("steps_taken_mean", "objective", "acquisition", "steps")