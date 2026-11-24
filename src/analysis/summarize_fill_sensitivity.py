"""Summarize scripts/run_fill_sensitivity.py: mean AUROC on the four positive controls
per arm, classifier and representation, plus PC1 vs number of filled values.
Writes results/sensitivity_fill/summary_fill_sensitivity.csv.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from metabo import config

# positive controls: 4 pairs x classifiers x (raw + 3 embeddings x k)
N_REC = 4 * len(config.MODELS) * (1 + 3 * len(config.EMBED_DIMS))

ARMS = ["baseline", "minfill", "wang", "perage"]
BASE = config.RESULTS_DIR / "sensitivity_fill"
rows = []
for arm in ARMS:
    recs = []
    for f in glob.glob(str(BASE / arm / "eval_*.json")):
        recs += json.load(open(f))
    d = pd.DataFrame(recs)
    d["pair"] = d.timepoint + "|" + d.target
    assert len(d) == N_REC, (arm, len(d), N_REC)
    views = {m: d[d.model == m] for m in config.MODELS}
    views["best3"] = (d.sort_values("auroc", ascending=False)
                       .groupby(["pair", "representation", "dim"]).head(1))
    for name, e in views.items():
        rows.append(dict(arm=arm, classifier=name, representation="raw", k_min=0, k_max=0,
                         auroc_low=e[e.representation == "raw"].auroc.mean(),
                         auroc_high=e[e.representation == "raw"].auroc.mean()))
        emb = e[e.representation != "raw"].groupby(["representation", "dim"]).auroc.mean()
        for rep in ["pca", "umap", "vae"]:
            rows.append(dict(arm=arm, classifier=name, representation=rep, k_min=2, k_max=16,
                             auroc_low=emb[rep].min(), auroc_high=emb[rep].max()))
out = pd.DataFrame(rows).round(3)
out.to_csv(BASE / "summary_fill_sensitivity.csv", index=False)
print(out[out.classifier.isin(["lr", "best3"])].to_string(index=False))

src = pd.read_csv(config.SOURCE_FILE,
                  sep=";", decimal=",", low_memory=False)
for arm in ARMS:
    for t, tt in [("6m", "6m"), ("18m", "18mth"), ("6y", "6y")]:
        X = pd.read_csv(config.DATA_DIR / "sensitivity_fill" / arm / f"metabolites_{t}_for_ml.csv",
                        index_col=0)
        n = src[src.TPn == tt][list(X.columns)].apply(pd.to_numeric, errors="coerce").isna().sum(axis=1)
        pc = PCA(1, random_state=42).fit_transform(RobustScaler().fit_transform(X))[:, 0]
        print(arm, t, "abs r(PC1, n filled) = %.2f" % abs(np.corrcoef(pc, n.values)[0, 1]))
