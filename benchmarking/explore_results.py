#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

pd.set_option("display.max_rows", 300)
matplotlib.rcParams.update({
    "figure.figsize": (12, 12),
    "axes.spines.top": False,
    "axes.spines.right": False
})
os.chdir(os.path.dirname(os.path.realpath(__file__)))

ensembles = pd.read_csv(os.path.join("deep_ensemble_test", "de_results.csv"), skipinitialspace=True)
mcdropout = pd.read_csv(os.path.join("mcdropout_test", "mc_results.csv"), skipinitialspace=True)
mcdropconnect = pd.read_csv(os.path.join("mcdropconnect_test", "mcdc_results.csv"), skipinitialspace=True)
gpr = pd.read_csv(os.path.join("gpr_test", "gpr_results.csv"), skipinitialspace=True)
der = pd.read_csv(os.path.join("der_test", "der_standard_results.csv"), skipinitialspace=True)
der_log = pd.read_csv(os.path.join("der_test", "der_log_results.csv"), skipinitialspace=True)

#ADD COLUMN MODEL NAME NEED TO REDO CSV's

# model_names = ["de", "mc", "mcdc", "gpr", "der_standard", "der_log"]
# dfs = [ensembles, mcdropout, mcdropconnect, gpr, der, der_log]
# filepaths = ["deep_ensembles_test", "mcdropout_test", "mcdropconnect_test", "gpr_test", "der_test", "der_log_test"]
# for model, df, filepath in zip(model_names, dfs, filepaths):
#     df.insert(0, "model", model)
#     df.to_csv(os.path.join(filepath, f"{model}_results.csv"))

#%%

DONT_GROUPBY = ["seed", "runtime", "true_min", "found_minimum", "converged"]

def check_convergence(results: pd.DataFrame, epsilon = 1e-2) -> pd.DataFrame:
    error = results["true_min"] - results["found_minimum"]
    converged = error.apply(lambda x: 1 if np.abs(x) <= epsilon else 0)
    results = results.assign(converged=converged)
    return results

def best_configs(results: pd.DataFrame, dont_groupby: list = DONT_GROUPBY, epsilon = 1e-2) -> pd.DataFrame:
    results = check_convergence(results)
    groupby = [col for col in results.columns if col not in dont_groupby]
    return results.groupby(by=groupby).mean().sort_values(by="converged", ascending=False)

def overall_best_config(results: pd.DataFrame, epsilon = 1e-2) -> pd.DataFrame:
    dont_groupby = DONT_GROUPBY + ["objective", "acquisition", "num_initial_points"]
    res = best_configs(results, dont_groupby = dont_groupby, epsilon=epsilon)
    return res

# %%
