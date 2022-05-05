#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
})
os.chdir(os.path.dirname(os.path.realpath(__file__)))

ensembles = pd.read_csv(os.path.join("deep_ensemble_test", "de_results.csv"), skipinitialspace=True)
# mcdropout = pd.read_csv(os.path.join("mcdropout_test", "mc_results.csv"), skipinitialspace=True)
gpr = pd.read_csv(os.path.join("gpr_test", "gpr_results.csv"), skipinitialspace=True)
der = pd.read_csv(os.path.join("der_test", "der_standard_results.csv"), skipinitialspace=True)

#%%

def check_convergence(results: pd.DataFrame, epsilon = 1e-2) -> pd.DataFrame:
    error = results["true_min"] - results["found_minimum"]
    converged = error.apply(lambda x: 1 if np.abs(x) <= epsilon else 0)
    results = results.assign(converged=converged)
    return results

def get_best_configs(results: pd.DataFrame, dont_groupby: list = ["seed", "runtime", "true_min", "found_minimum", "converged"], epsilon = 1e-2) -> pd.DataFrame:
    results = check_convergence(results)
    groupby = [col for col in results.columns if col not in dont_groupby]
    return results.groupby(by=groupby).mean().sort_values(by="converged", ascending=False)

der_group = get_best_configs(der)
# %%
