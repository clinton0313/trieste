#%%
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from .plotting_params import *
from typing import Callable

matplotlib.rcParams.update({
    "figure.figxize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False,
})
matplotlib.style.use("seaborn-bright")
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def nlpd(
    mean_pred: np.ndarray,
    var_pred: np.ndarray,
    query_points: np.ndarray, 
    objective: Callable
) -> np.ndarray:
    true_points = objective(query_points)
    return (
        (mean_pred - true_points)**2 / (2 * var_pred)
        + 1/2 * np.log(2 * np.math.pi * var_pred)
    )
