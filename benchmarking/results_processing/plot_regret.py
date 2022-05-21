#%%

import itertools
import matplotlib
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from utils.get_regret import (
    plot_min_regret_model_comparison,
    simple_regret
)
from utils.plotting_params import (
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
SAVEPATH = os.path.join("figs", "regret_plots", "linear_scale", "simple_regret")
PREFIX = "simple_regret"
os.makedirs(SAVEPATH, exist_ok=True)
for obj, acq in tqdm(itertools.product(OBJECTIVE_DICT.keys(), ACQUISITION_DICT.keys())):
    f, a = plt.subplots()
    #Plotting simple regret (log_scale True by default)
    plot_min_regret_model_comparison(
        obj, 
        acq, 
        ax=a, 
        xlim=(0, 500), 
        average_regret_kwargs = {
            "regret_function": simple_regret,
            "regret_name": "simple_regret",
        },
        log_scale = False,
        n_stds = 0,
        alpha = 0.5,
    )
    # # Plotting min regret with log scale True by default
    # plot_min_regret_model_comparison(
    #     obj, 
    #     acq, 
    #     ax=a, 
    #     xlim=(0, 500), 
    #     log_scale = False,
    #     n_stds = 0
    # )
    f.savefig(
        os.path.join(SAVEPATH, f"{PREFIX}_{obj}_{acq}_model_comparison.png"),
        facecolor="white",
        transparent=False
    )
    f.clear()
    plt.close(f)

# %%
