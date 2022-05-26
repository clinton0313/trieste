#%%

import itertools
import matplotlib
from matplotlib import tight_layout
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

# make_plots("simple_regret")
# make_plots("min_regret")
# make_plots("overlay")

def plot_side_by_side(
    acquisitions: list,
    objectives: list,
    nrows: int,
    ncols: int,
    figsize: tuple,
    ylims: list = [],
    savepath: str = os.path.join("figs", "regret_plots", "sidebyside"),
    **plot_kwargs
) -> None:
    os.makedirs(savepath, exist_ok=True)
    fig, axes = plt.subplots(nrows, ncols, figsize = figsize, tight_layout=True)
    if ylims == []:
        ylims = [plot_kwargs.get("ylim", None) for _ in range(len(axes))]
    for (obj, acq), ax, ylim in zip(itertools.product(objectives, acquisitions), axes.ravel(), ylims):
        plot_kwargs.update({"ylim": ylim})
        plot_min_regret_model_comparison(obj, acq, ax, legend = False, **plot_kwargs)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.supxlabel("Function Evaluations")
    fig.supylabel("Cumulative Minimum Regret")
    fig.savefig(
        os.path.join(savepath, f"{'_'.join(objectives + acquisitions)}_min_regret.png"),
        facecolor="white",
        transparent=True
    )

plot_kwargs = {
    "log_scale": True,
    "n_stds": 1,
    "ylabel": "",
    "xlabel": ""
}

fig_1 = {
    "acquisitions": ["ts"],
    "objectives": ["noisy_shekel4", "noisy_ackley5", "noisy_hartmann6"],
    "nrows": 1,
    "ncols": 3,
    "figsize": (18, 6),
    "average_regret_kwargs": {
        "max_steps": 5000
    },
    "xlim": (0, 5000),
    "ylims": [(3e-1, 1),(1e-2, 1),(4e-2,1)]
}
# shekel: 2e-1, ackley: ?, hartmann: 3e-2

fig_2 = {
    "acquisitions": ["ei", "ts"],
    "objectives": ["hartmann3"],
    "nrows": 1,
    "ncols": 2,
    "figsize": (12, 6),
    "average_regret_kwargs": {
        "max_steps": 2000
    },
    "xlim": (0, 500),
    "ylim": (6e-5,1)
}


fig_3 = {
    "acquisitions": ["ei", "ts"],
    "objectives": ["goldstein2"],
    "nrows": 1,
    "ncols": 2,
    "figsize": (12, 6),
    "average_regret_kwargs": {
        "max_steps": 2000
    },
    "xlim": (0, 500),
    "ylim": (1e-4,1)
}

fig_4 = {
    "acquisitions": ["ei"],
    "objectives": ["shekel4", "ackley5", "hartmann6"],
    "nrows": 1,
    "ncols": 3,
    "figsize": (18, 6),
    "average_regret_kwargs": {
        "max_steps": 2000
    },
    "xlim": (0, 500),
    "ylims": [(3e-2, 1), (2e-1, 1), (3e-4,1)]
}
# shekel: 9e-2, ackley: 2e-2, hartmann: 6e-4

fig_5 = {
    "acquisitions": ["ts"],
    "objectives": ["shekel4", "ackley5", "hartmann6"],
    "nrows": 1,
    "ncols": 3,
    "figsize": (18, 6),
    "average_regret_kwargs": {
        "max_steps": 2000
    },
    "xlim": (0, 500),
    "ylims": [(3e-1, 1),(3e-1, 1),(3e-2,1)]
}
# shekel: 7e-1, ackley5: 6e-1, hartmann: 8e-2

fig_6 = {
    "acquisitions": ["ei", "ts"],
    "objectives": ["dropw2"],
    "nrows": 1,
    "ncols": 2,
    "figsize": (12, 6),
    "average_regret_kwargs": {
        "max_steps": 2000
    },
    "xlim": (0, 500),
    "ylim": (6e-3, 1)
}

fig_7 = {
    "acquisitions": ["ei", "ts"],
    "objectives": [
        "michal2", 
        "scaled_branin", 
        "goldstein2", 
        "dropw2", 
        "eggho2", 
        "hartmann3", 
        "rosenbrock4"
    ],
    "nrows": 4,
    "ncols": 4,
    "figsize": (24, 24),
    "average_regret_kwargs": {
        "max_steps": 2000,
    },
    "xlim": (0, 500),
    "ylim": (1e-4, 1)
}

figs = [fig_1, fig_2, fig_3, fig_4, fig_5, fig_6, fig_7]
for fig_args in tqdm(figs):
    fig_args.update(plot_kwargs)
    plot_side_by_side(**fig_args)