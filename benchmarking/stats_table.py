# %%
# from benchmarking.benchmarking_utils import *
import pandas as pd
import numpy as np
import os, json
from glob import glob
# %%
PATH = "trieste\\benchmarking\\results"
COLUMNS = [
    "model", 
    "objective", 
    "acquisition", 
    "seed", 
    "true_min", 
    "found_minimum", 
    "steps_taken",  
    "optimize_runtime"
]

csvs = [
    file for path, _, _ in os.walk(PATH) for file in glob(os.path.join(path, "*.csv"))
]
with open('trieste\\benchmarking\\output_ranges.json') as f:
    output_ranges = json.load(f)

# %%
dfs = []
for file in csvs:
    df = pd.read_csv(file, usecols=COLUMNS)

    df = (
        df.assign(
            optimize_runtime_step = df["optimize_runtime"] / (df["steps_taken"] + 1),
            final_regret = np.abs(df["found_minimum"] - df["true_min"]) / df["objective"].map(output_ranges),
            found_min = lambda dtf: np.where(dtf.final_regret < 1e-03, 1, 0)
        )
    )
    dfs.append(df)

df = pd.concat(dfs)
# %%
df = df[["model", "objective", "acquisition", "optimize_runtime_step", "steps_taken", "final_regret", "found_min"]]

stats = (
    df
    .groupby(["objective", "model", "acquisition"])
    .describe()
    .loc[:, lambda df: df.columns.get_level_values(1).isin({"mean", "std"})]
)

stats.loc[:, lambda df: df.columns.get_level_values(1)=="mean"] = (
    stats.loc[:, lambda df: df.columns.get_level_values(1)=="mean"].round(2)
)

stats.loc[:, lambda df: df.columns.get_level_values(1)=="std"] = (
    stats.loc[:, lambda df: df.columns.get_level_values(1)=="std"].round(2)
)

stats = stats.drop(columns=[("found_min", "std"), ("final_regret", "std")])

stats.to_csv("trieste\\benchmarking\\stats.csv")
# %%
