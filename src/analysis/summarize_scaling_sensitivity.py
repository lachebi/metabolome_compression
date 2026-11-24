"""Summarize scripts/run_scaling_sensitivity.py against the baseline arm of the fill
sensitivity (same software, inputs and folds): mean AUROC on the four positive controls
per classifier and representation, plus PC1 vs number of filled values.
Writes results/sensitivity_scaling/summary_scaling_sensitivity.csv.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from metabo import config

# positive controls: 4 pairs x classifiers x (raw + 3 embeddings x k)
N_REC = 4 * len(config.MODELS) * (1 + 3 * len(config.EMBED_DIMS))

ARMS = {"robust": config.RESULTS_DIR / "sensitivity_fill" / "baseline",
        "standard": config.RESULTS_DIR / "sensitivity_scaling" / "standard"}
rows = []
for arm, folder in ARMS.items():
    d = pd.DataFrame([r for f in glob.glob(str(folder / "eval_*.json")) for r in json.load(open(f))])
    assert len(d) == N_REC, (arm, len(d), N_REC)
    if arm == "standard":
        assert set(d.scaler) == {"StandardScaler"}
    d["pair"] = d.timepoint + "|" + d.target
    views = {m: d[d.model == m] for m in config.MODELS}
    views["best3"] = d.sort_values("auroc", ascending=False).groupby(["pair", "representation", "dim"]).head(1)
    for name, e in views.items():
        by = e.groupby(["representation", "dim"]).auroc.mean()
        rows.append(dict(arm=arm, classifier=name, representation="raw", low=by["raw"].iloc[0], high=by["raw"].iloc[0]))
        for rep in ["pca", "umap", "vae"]:
            rows.append(dict(arm=arm, classifier=name, representation=rep, low=by[rep].min(), high=by[rep].max()))
out = pd.DataFrame(rows).round(3)
out.to_csv(config.RESULTS_DIR / "sensitivity_scaling" / "summary_scaling_sensitivity.csv", index=False)
print(out[out.classifier.isin(["lr", "best3"])].to_string(index=False))

lr = {a: pd.DataFrame([r for f in glob.glob(str(p / "eval_*.json")) for r in json.load(open(f))]) for a, p in ARMS.items()}
for a, d in lr.items():
    print(a, "per pair LR raw:", d[(d.model == "lr") & (d.representation == "raw")].set_index(["timepoint", "target"]).auroc.round(3).to_dict())
    e = d[(d.model == "lr") & (d.representation != "raw")].groupby(["timepoint", "target"]).auroc.agg(["mean", "max"]).round(3)
    print(e.to_string())

src = pd.read_csv(config.SOURCE_FILE, sep=";", decimal=",", low_memory=False)
for t, tt in [("6m", "6m"), ("18m", "18mth"), ("6y", "6y")]:
    X = pd.read_csv(config.METAB_FILES[t], index_col=0)
    n = src[src.TPn == tt][list(X.columns)].apply(pd.to_numeric, errors="coerce").isna().sum(axis=1).values
    res = {}
    for name, S in [("robust", RobustScaler), ("standard", StandardScaler)]:
        pc = PCA(1, random_state=42).fit_transform(S().fit_transform(X))[:, 0]
        res[name] = round(abs(np.corrcoef(pc, n)[0, 1]), 2)
    print(t, "abs r(PC1, n filled)", res)
