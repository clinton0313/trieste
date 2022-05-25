
#%%
import numpy as np
import os
import pandas as pd

from utils.plotting_params import MODELS_DICT, RESULTS_DIR

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%%
if __name__ == "__main__":
    nlpd = pd.read_csv(os.path.join(RESULTS_DIR, "nlpd_results.csv"), skipinitialspace=True)
    model_dfs = []
    for model, model_meta in MODELS_DICT.items():
        if model != "random":
            df = pd.read_csv(os.path.join(RESULTS_DIR, model, f"{model_meta['name']}_results.csv"), skipinitialspace=True)
            df = df.groupby(by=["objective", "acquisition"]).mean().reset_index()
            df["model"] = model_meta["name"]
            df["final_regret"] = df["found_minimum"] - df["true_min"]
            model_dfs.append(df)

    model_dfs = pd.concat(model_dfs, ignore_index=True)

    all_data = pd.merge(nlpd, model_dfs.loc[:, ["model", "objective", "acquisition", "final_regret"]], on=["model", "objective", "acquisition"])

    print("Correlations:")
    print(f"All Models")
    print(all_data.loc[:,["nlpd_end_mean", "nlpd_minimizer_end_mean", "final_regret"]].corr()["final_regret"])

    for model_meta in MODELS_DICT.values():
        filtered = all_data[all_data["model"] == model_meta["name"]]
        print(f"Model: {model_meta['name']}")
        print(filtered.loc[:,["nlpd_end_mean", "nlpd_minimizer_end_mean", "final_regret"]].corr()["final_regret"])

# %%
