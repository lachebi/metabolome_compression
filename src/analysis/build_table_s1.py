"""Table S1: every outcome x timepoint x representation x k.

Per cell: the best of three classifiers (with CI, name, AUPRC and calibration), each
classifier on its own (with CI and calibration), and the permutation result on the raw
metabolome with BH q-values over all pairs. Reads result files only.
Writes results/supplement_S1_auroc_all_pairs.csv.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metabo import config
from plot_figures import nice

R = config.RESULTS_DIR
ev = pd.DataFrame([r for f in glob.glob(str(R / "eval_*.json")) for r in json.load(open(f))]).drop_duplicates("key")
ev = ev[ev.target.isin(config.OUTCOMES)]
pm = pd.concat([pd.DataFrame(json.load(open(f))) for f in glob.glob(str(R / "permutation_*.json"))])
pm = pm[pm.target.isin(config.OUTCOMES)]
pm = pm[pm.n_perm == pm.groupby("key").n_perm.transform("max")].drop_duplicates("key")
pm = pm[["timepoint", "target", "obs_auroc", "null_mean", "p_value"]].reset_index(drop=True)

n = len(pm)
assert n == len(config.OUTCOMES) * len(config.TIMEPOINTS), n
order = pm.p_value.values.argsort()
q = pm.p_value.values[order] * n / np.arange(1, n + 1)
q = np.minimum.accumulate(q[::-1])[::-1]
pm["raw_perm_q_bh42"] = np.nan
pm.iloc[order, pm.columns.get_loc("raw_perm_q_bh42")] = np.minimum(q, 1)

k = ["timepoint", "target", "representation", "dim"]
s = ev.loc[ev.groupby(k).auroc.idxmax(), k + ["prevalence", "model", "auroc", "auroc_lo", "auroc_hi", "auprc",
                                               "cal_intercept", "cal_slope", "brier"]].rename(
    columns={"model": "best_model", "auroc": "auroc_best_of_3", "auroc_lo": "best_lo", "auroc_hi": "best_hi",
             "auprc": "auprc_best", "cal_intercept": "cal_intercept_best", "cal_slope": "cal_slope_best",
             "brier": "brier_best"})
for model, name in [("lr", "lr"), ("rfc", "rf"), ("knn", "knn")]:
    one = ev[ev.model == model][k + ["auroc", "auroc_lo", "auroc_hi", "cal_intercept", "cal_slope", "brier"]].rename(
        columns={"auroc": f"auroc_{name}", "auroc_lo": f"auroc_{name}_lo", "auroc_hi": f"auroc_{name}_hi",
                 "cal_intercept": f"cal_intercept_{name}", "cal_slope": f"cal_slope_{name}", "brier": f"brier_{name}"})
    s = s.merge(one, on=k, validate="one_to_one")
s = s.merge(pm.rename(columns={"obs_auroc": "raw_perm_auroc", "null_mean": "raw_perm_null_mean",
                               "p_value": "raw_perm_p"}), on=["timepoint", "target"])
s["nominal_p05"] = s.raw_perm_p < 0.05
s.insert(1, "outcome", s.target.map(nice))
s = s.drop(columns="target").sort_values(["outcome", "timepoint", "representation", "dim"])

# five decimals for p, so a value just below 0.05 does not print as 0.05
s = s.round(4).assign(raw_perm_p=s.raw_perm_p.round(5))
s.to_csv(R / "supplement_S1_auroc_all_pairs.csv", index=False)
print("rows", len(s), "pairs", s.groupby(["timepoint", "outcome"]).ngroups,
      "p < 0.05:", int(s[s.representation == "raw"].nominal_p05.sum()))
