#%%

import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from utils.plotting_params import *
from plot_success import quick_bar_plots
from typing import Callable, Tuple

matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False,
})
matplotlib.style.use("seaborn-bright")
os.chdir(os.path.dirname(os.path.realpath(__file__)))


if __name__ == "__main__":
    nlpd_res = pd.read_csv(os.path.join(RESULTS_DIR, "nlpd_results.csv"))

    for step in ["start", "end", "difference"]:
        nlpd_res[f"nlpd_{step}_total"] = nlpd_res[f"nlpd_{step}_mean"] + nlpd_res[f"nlpd_minimizer_{step}_mean"]

    for t in [
        "start_total",
        "end_total",
        "start_mean",
        "end_mean",
        "minimizer_start_mean",
        "minimizer_end_mean",
        "difference_mean",
        "difference_total"
    ]:
        quick_bar_plots(
            x = "model",
            y = f"nlpd_{t}",
            hue = "acquisition",
            plot_by = "objective",
            title = t,
            stats = nlpd_res,
            savepath = os.path.join("figs", "nlpd", "log_scale"),
            fig_rows = 4,
            fig_cols = 3,
            log_scale = True
        )
        quick_bar_plots(
            x = "model",
            y = f"nlpd_{t}",
            hue = "acquisition",
            plot_by = "objective",
            title = t,
            stats = nlpd_res,
            savepath = os.path.join("figs", "nlpd", "linear_scale"),
            fig_rows= 4,
            fig_cols = 3,
            log_scale = False
        )

#%%