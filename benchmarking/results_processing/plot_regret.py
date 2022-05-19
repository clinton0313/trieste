#%%

import itertools
import matplotlib
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from utils.get_regret import (
    plot_min_regret_model_comparison,
    OBJECTIVE_DICT,
    ACQUISITION_DICT
)

matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
})
matplotlib.style.use("seaborn-bright")
matplotlib.use("tkagg")

os.chdir(os.path.dirname(os.path.realpath(__file__)))

for obj, acq in tqdm(itertools.product(OBJECTIVE_DICT.keys(), ACQUISITION_DICT.keys())):
    f, a = plt.subplots()
    plot_min_regret_model_comparison(obj, acq, ax=a, xlim=(0, 500))
    f.savefig(
        os.path.join("figs", f"{obj}_{acq}_model_comparison.png"),
        facecolor="white",
        transparent=False
    )
    f.clear()
    plt.close(f)
