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
os.makedirs(os.path.join("figs", "success"), exist_ok=True)

stats = pd.read_csv("stats_flattened.csv")

#%%

for model in stats["model"].unique():
    f, a = plt.subplots()
    sns.barplot("objective", "found_min", hue="acquisition", data=stats[stats["model"]==model], ax=a)
    a.set_title(f"{model}")
    f.savefig(os.path.join("figs", "success", f"{model}_success.png"), facecolor="white", transparent=False)
    f.clear()
    plt.close(f)

f, a = plt.subplots()
sns.barplot("objective", "found_min", hue="acquisition", data=stats, ax=a)
a.set_title("Overall")
f.savefig(os.path.join("figs", "success", f"overall_success.png"), facecolor="white", transparent=False)

f, ax = plt.subplots(3, 2, tight_layout=True)
sns.barplot("objective", "found_min", hue="acquisition", data=stats, ax=ax.ravel()[0])
ax.ravel()[0].set_xticklabels(ax.ravel()[0].get_xticklabels(), rotation=45, horizontalalignment='right')
ax.ravel()[0].set_title("Overall")
for model, a in zip(stats["model"].unique(), ax.ravel()[1:]):
    sns.barplot("objective", "found_min", hue="acquisition", data=stats[stats["model"]==model], ax=a)
    a.set_title(f"{model}")
    a.set_xticklabels(a.get_xticklabels(), rotation=45, horizontalalignment='right')
f.savefig(os.path.join("figs", "success", f"all_success.png"), facecolor="white", transparent=False)