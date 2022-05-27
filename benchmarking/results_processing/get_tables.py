# %%

import pickle, os
from pathlib import Path
from benchmarking.results_processing.utils.plotting_params import (
    COLOR_DICT, 
    MODELS_DICT, 
    OBJECTIVE_DICT,
)
from benchmarking.results_processing.utils.get_regret import average_regret
import numpy as np
import pandas as pd
from glob import glob
import json

COLUMNS = [
    "model", 
    "objective", 
    "acquisition", 
    "true_min", 
    "found_minimum", 
    "steps_taken",  
    "optimize_runtime",
]

NOISELESS_FUNCTIONS = [
    'scaled_branin', 
    'dropw2', 
    'eggho2', 
    'goldstein2', 
    'michal2',
    'hartmann3', 
    'rosenbrock4',
    'shekel4', 
    'hartmann6', 
    'ackley5'
]

NOISY_FUNCTIONS = [
    'noisy_shekel4', 
    'noisy_hartmann6', 
    'noisy_ackley5'
]

ORDER_NOISELESS = [
    'Deep Ensembles', 
    'Deep Evidential', 
    'Direct Epistemic',
    'MC Dropout',
    'GPR',
]
ORDER_NOISY = [
    'Deep Ensembles', 
    'Deep Evidential', 
    'Direct Epistemic',
    'MC Dropout',
    'SVGP',
    'GPR'
]

ORDER_FUNCTIONS_NOISELESS = [
    'Log Goldstein-Price-2',
    'Scaled Branin-2',
    'Michalwicz-2',
    'Dropwave-2',
    'Eggholder-2',
    'Hartmann-3', 
    'Rosenbrock-4',
    'Shekel-4',
    'Ackley-5', 
    'Hartmann-6',
]

NOISY_NAME_MAP = {
    'Noisy Shekel-4': "Shekel-4",
    'Noisy Ackley-5': "Ackley-5", 
    'Noisy Hartmann-6': "Hartmann-6" 
}

ORDER_FUNCTIONS_NOISY = [
    'Noisy Shekel-4',
    'Noisy Ackley-5', 
    'Noisy Hartmann-6', 
]

basedir = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
# %%
csvs = [
    file for path, _, _ in os.walk(basedir / "results") for file in glob(os.path.join(path, "*.csv")) 
    if "noisy" not in file 
    if "nlpd" not in file
    if "random" not in file
]

# %%
with open(basedir / 'output_ranges.json') as f:
    output_ranges = json.load(f)

def produce_df(files, objectives: list):
    dfs = []
    for file in files:
        df = pd.read_csv(file, usecols=COLUMNS)
        df = (
            df.assign(
                optimize_runtime_step = df["optimize_runtime"] / (df["steps_taken"] + 1),
                final_regret = np.abs(df["found_minimum"] - df["true_min"]) / df["objective"].map(output_ranges),
                found_min = lambda dtf: np.where(dtf.final_regret < 1e-03, 1, 0),
                optimize_runtime = lambda dtf: dtf.optimize_runtime / 60
            )
        )
        dfs.append(df)

    df = pd.concat(dfs)
    df = (
        df[["model", "objective", "acquisition", "optimize_runtime", "optimize_runtime_step", "steps_taken", "final_regret", "found_min"]]
        .loc[df.objective.isin(objectives)]
    )

    return df

def produce_regrets(files):
    regrets = []
    for file in files:
        regret = average_regret(pd.read_csv(file), Path(file).parents[0])
        regret = regret.loc[regret.objective.str.contains("noisy")]
        # regret["final_regret"] = regret.mean_min_regret.apply(lambda row: np.min(row))
        # regrets.append(regret[["model", "objective", "final_regret"]])
        regrets.append(regret)

    regrets = pd.concat(regrets)
    return regrets

def produce_latex(
    dataframe: pd.DataFrame, 
    target: str = "optimize_runtime_step", 
    decimals: int = 2,
    noise: bool = False
):
    tmp_ = (
        dataframe
        .groupby(["model", "objective", "acquisition"])
        [["optimize_runtime_step", "optimize_runtime", "final_regret"]]
        .agg("mean")
        .reset_index()
    )
    if target == "final_regret" and noise:
        regrets = produce_regrets(csvs)
        tmp_ = pd.merge(
            tmp_[["model", "objective", "acquisition", "optimize_runtime_step", "optimize_runtime"]], 
            regrets,
            left_on=["model", "objective"], 
            right_on=["model", "objective"],
            how="left"
        )

    tmp_["objective"] = tmp_["objective"].map(OBJECTIVE_DICT)
    tmp_["model"] = tmp_["model"].map({MODELS_DICT[k]["name"]: MODELS_DICT[k]["label"] for k in MODELS_DICT.keys()})
    tmp_ = pd.pivot_table(
        tmp_, 
        columns=["acquisition", "model"], 
        index=["objective"], 
        values=target
    ).round(decimals)

    if noise:
        title = "noisy"
        tmp_ = tmp_[list(zip(tmp_.columns.get_level_values(0), ORDER_NOISY))]
        tmp_ = tmp_.reindex(ORDER_FUNCTIONS_NOISY)
        tmp_.index = tmp_.index.map(NOISY_NAME_MAP)

    else:
        title = "noiseless"
        tmp_ = tmp_[list(zip(tmp_.columns.get_level_values(0), ORDER_NOISELESS*2))]
        tmp_ = tmp_.reindex(ORDER_FUNCTIONS_NOISELESS)


    if target == "final_regret":
        tmp_ = tmp_.astype(str)
        tmp_.replace({"0.0": "-"}, inplace=True)

    tmp_.to_latex(basedir / "results_processing" / "tables" / f"{title}_{target}_results.tex")
    
    return tmp_

# %%
noiseless = produce_df(csvs, NOISELESS_FUNCTIONS)
noisy = produce_df(csvs, NOISY_FUNCTIONS)

noiseless_table = produce_latex(noiseless, target="final_regret", decimals=2, noise=False)
noisy_table = produce_latex(noisy, target="final_regret", decimals=2, noise=True)
# %%
# %%
regrets = produce_regrets(csvs)
# %%
