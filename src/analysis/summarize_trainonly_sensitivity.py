"""Summarize scripts/run_trainonly_sensitivity.py against the baseline arm of the fill
sensitivity: mean AUROC on the four positive controls per classifier and representation,
and per pair for logistic regression.
Writes results/sensitivity_trainonly/summary_trainonly_sensitivity.csv.
"""
import glob
import json

import pandas as pd

from metabo import config

# positive controls: 4 pairs x classifiers x (raw + 3 embeddings x k)
N_REC = 4 * len(config.MODELS) * (1 + 3 * len(config.EMBED_DIMS))

ARMS = {"baseline": config.RESULTS_DIR / "sensitivity_fill" / "baseline",
        "fill": config.RESULTS_DIR / "sensitivity_trainonly" / "fill",
        "fill_filter": config.RESULTS_DIR / "sensitivity_trainonly" / "fill_filter"}
data, rows = {}, []
for arm, folder in ARMS.items():
    d = pd.DataFrame([r for f in glob.glob(str(folder / "eval_*.json")) for r in json.load(open(f))])
    assert len(d) == N_REC, (arm, len(d), N_REC)
    d["pair"] = d.timepoint + "|" + d.target
    data[arm] = d
    views = {m: d[d.model == m] for m in config.MODELS}
    views["best3"] = d.sort_values("auroc", ascending=False).groupby(["pair", "representation", "dim"]).head(1)
    for name, e in views.items():
        by = e.groupby(["representation", "dim"]).auroc.mean()
        rows.append(dict(arm=arm, classifier=name, representation="raw", low=by["raw"].iloc[0], high=by["raw"].iloc[0]))
        for rep in ["pca", "umap", "vae"]:
            rows.append(dict(arm=arm, classifier=name, representation=rep, low=by[rep].min(), high=by[rep].max()))
out = pd.DataFrame(rows).round(3)
out.to_csv(config.RESULTS_DIR / "sensitivity_trainonly" / "summary_trainonly_sensitivity.csv", index=False)
print(out[out.classifier.isin(["lr", "best3"])].to_string(index=False))
keys = ["pair", "representation", "dim", "model"]
base = data["baseline"].set_index(keys).auroc
for arm in ["fill", "fill_filter"]:
    diff = data[arm].set_index(keys).auroc - base
    for rep, g in [("raw and PCA", diff[diff.index.get_level_values("representation").isin(["raw", "pca"])]),
                   ("UMAP", diff[diff.index.get_level_values("representation") == "umap"]),
                   ("VAE", diff[diff.index.get_level_values("representation") == "vae"])]:
        print(arm, rep, "arm minus baseline: mean %+.3f, max |diff| %.3f" % (g.mean(), g.abs().max()))
    lr = data[arm][data[arm].model == "lr"]
    print(arm, "LR raw per pair", lr[lr.representation == "raw"].set_index("pair").auroc.round(3).to_dict())
    print(lr[lr.representation != "raw"].groupby("pair").auroc.max().round(3).to_dict())
lr = data["baseline"][data["baseline"].model == "lr"]
print("baseline LR raw", lr[lr.representation == "raw"].set_index("pair").auroc.round(3).to_dict())
print(lr[lr.representation != "raw"].groupby("pair").auroc.max().round(3).to_dict())
