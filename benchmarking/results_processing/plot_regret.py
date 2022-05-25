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
SAVEPATH = os.path.join("figs", "regret_plots", "log_scale")

def make_plots(kind: str, savepath: str = os.path.join("figs", "regret_plots", "log_scale")):
    '''Quick and dirty wrapper here.'''
    savepath = os.path.join(savepath, kind)
    os.makedirs(savepath, exist_ok=True)
    for obj, acq in tqdm(itertools.product(OBJECTIVE_DICT.keys(), ACQUISITION_DICT.keys())):
        if "noisy" in obj:
            max_steps = 5100
            xlim = (0, 5000)
        else:
            max_steps = 2100
            xlim = (0, 500)        
        
        f, a = plt.subplots()
        # Plotting simple regret (log_scale True by default)
        if kind =="overlay" or kind =="simple_regret":
            plot_min_regret_model_comparison(
                obj, 
                acq, 
                ax=a, 
                xlim=xlim, 
                average_regret_kwargs = {
                    "regret_function": simple_regret,
                    "regret_name": "simple_regret",
                    "max_steps": max_steps
                },
                log_scale = True,
                n_stds = 0,
                alpha = 0.5,
            )
        # Plotting min regret with log scale True by default
        if kind == "overlay" or kind =="min_regret":
            plot_min_regret_model_comparison(
                obj, 
                acq, 
                ax=a, 
                xlim=xlim, 
                average_regret_kwargs={
                    "max_steps": max_steps
                },
                log_scale = True,
                n_stds = 1
            )
        f.savefig(
            os.path.join(savepath, f"{kind}_{obj}_{acq}_model_comparison.png"),
            facecolor="white",
            transparent=False
        )
        f.clear()
        plt.close(f)

# %%

make_plots("simple_regret")
make_plots("min_regret")
make_plots("overlay")