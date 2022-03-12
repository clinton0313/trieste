#%%

import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

os.chdir(os.path.dirname(os.path.realpath(__file__)))
#%%

def plot_data(data, rate):
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    rate_filter = data[data["rate"] == rate]

    fig, ax = plt.subplots(1, 2, figsize=(20, 10), tight_layout=True)
    fig.suptitle(f"Rate = {rate}")
    for n, color in zip(data["num_hidden_layers"].unique(), colors):

        for a, y_label in zip(ax, ["mae", "time"]):
            a.plot("units", y_label, data=rate_filter[rate_filter["num_hidden_layers"] == n], color=color, label=f"{n} layers")
            a.set_ylabel(y_label.capitalize())
            a.set_xlabel("Units")
            a.legend()
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        ax[0].axhline(4, color="black", linestyle="dashed")
        ax[0].axhline(2, color="red", linestyle="dashed")
        ax[0].set_ylim(0, 20)

        ax[1].axhline(100, color="red", linestyle="dashed")
        ax[1].set_ylim(0, 300)

    return fig

def save_figs(results, savepath, suffix="dropconnect"):
    data = results[results["dropout"] == suffix]
    for rate in data.rate.unique():
        fig = plot_data(data, rate)
        fig.savefig(
            os.path.join(savepath, f"{suffix}_{str(rate).replace('.', '_')}.png"), 
            facecolor="white",
            transparent=False
        )
#%%

with open("results.pkl", "rb") as infile:
    results = pickle.load(infile)

records = [result["params"] for result in results]
for record, result in zip(records, results):
    record.update({"mae":result["mae"], "time": result["time"]})

data = pd.DataFrame.from_records(records)
data = data[data["activation"] == "relu"]

#%%
save_figs(data, "figs", suffix="dropconnect")
save_figs(data, "figs", suffix="standard")
# %%
