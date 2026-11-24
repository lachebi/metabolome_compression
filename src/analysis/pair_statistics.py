"""Statistics from the result files; no model is fitted. Writes results/stats/:
  comparison_sets.csv     mean AUROC per set of pairs, classifier, representation and k
  per_outcome_means.csv   the same with each outcome counted once
  bh_over_42.csv          permutation p and BH q over all pairs (42 in the paper)
  wilcoxon.csv            raw minus embedding, LR, across outcomes and across pairs
  calibration.csv, calibration_by_set.csv   intercept, slope and Brier
  counts.csv              configurations below 0.5, vitamin D means, null means
Sets: all pairs, the positive controls (sex at three ages, fish oil at 6 months) and the
pairs with permutation p < 0.05.
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from metabo import config
from metabo.stats import bh

R = config.RESULTS_DIR
OUT = R / "stats"
OUT.mkdir(exist_ok=True)
ROSTER = config.OUTCOMES
N_PAIRS = len(ROSTER) * len(config.TIMEPOINTS)
CONTROLS = {("6m", "sex"), ("18m", "sex"), ("6y", "sex"), ("6m", "fishoil")}
REPS = ["pca", "umap", "vae"]

ev = pd.DataFrame([r for f in sorted(glob.glob(str(R / "eval_*.json"))) for r in json.load(open(f))])
ev = ev.drop_duplicates("key")
ev = ev[ev.target.isin(ROSTER)].copy()
# REPS (the 3 embeddings; "raw" is the separate "1 +" term) is this file's own independent
# list, not derived from ev itself, so a wholly missing representation makes this assertion
# fail rather than silently shrinking its own expectation to match (ev.representation.nunique()
# would count "raw" too, so the previous version used (nunique - 1); using the fixed list
# directly needs no "- 1" since REPS never included "raw" to begin with).
n_rec = N_PAIRS * len(config.MODELS) * (1 + len(REPS) * len(config.EMBED_DIMS))
assert len(ev) == n_rec, (len(ev), n_rec)
pm = pd.concat([pd.DataFrame(json.load(open(f))) for f in sorted(glob.glob(str(R / "permutation_*.json")))])
pm = pm[pm.target.isin(ROSTER)]
pm = pm[pm.n_perm == pm.groupby("key").n_perm.transform("max")].drop_duplicates("key")
pm = pm.sort_values(["timepoint", "target"]).reset_index(drop=True)
assert len(pm) == N_PAIRS, len(pm)
pm["q_bh42"] = bh(pm.p_value)
pm[["timepoint", "target", "obs_auroc", "null_mean", "p_value", "q_bh42"]].sort_values("p_value").to_csv(
    OUT / "bh_over_42.csv", index=False)
predictive = set(zip(pm[pm.p_value < .05].timepoint, pm[pm.p_value < .05].target))

key = ["timepoint", "target", "representation", "dim"]
best = ev.loc[ev.groupby(key).auroc.idxmax()].assign(model="best3")
views = pd.concat([ev, best])
pair = list(zip(views.timepoint, views.target))
sets = {"all": [True] * len(views), "controls": [p in CONTROLS for p in pair], "p05": [p in predictive for p in pair]}
rows, rows_o = [], []
for name, mask in sets.items():
    v = views[mask]
    for (m, rep, dim), g in v.groupby(["model", "representation", "dim"]):
        rows.append(dict(set=name, classifier=m, representation=rep, k=dim, n_pairs=len(g), mean_auroc=g.auroc.mean()))
        rows_o.append(dict(set=name, classifier=m, representation=rep, k=dim, n_outcomes=g.target.nunique(),
                           mean_auroc=g.groupby("target").auroc.mean().mean()))
comp = pd.DataFrame(rows)
comp.to_csv(OUT / "comparison_sets.csv", index=False)
pd.DataFrame(rows_o).to_csv(OUT / "per_outcome_means.csv", index=False)

lr = ev[ev.model == "lr"]
raw = lr[lr.representation == "raw"].set_index(["timepoint", "target"]).auroc
rows = []
for rep in REPS:
    emb = lr[lr.representation == rep].groupby(["timepoint", "target"]).auroc.mean()
    diff = (raw - emb).dropna()
    assert len(diff) == N_PAIRS
    by_out = diff.groupby("target").mean()
    # unrounded values: wilcoxon drops exact zeros
    for level, d in [("outcomes", by_out), ("pairs", diff)]:
        rows.append(dict(representation=rep, level=level, n=len(d), mean_diff=d.mean(),
                         n_raw_higher=int((d > 0).sum()), p=wilcoxon(d.values).pvalue))
pd.DataFrame(rows).to_csv(OUT / "wilcoxon.csv", index=False)

cal = (best.assign(kind=np.where(best.representation == "raw", "raw", "embedded"))
       .groupby("kind")[["cal_intercept", "cal_slope", "brier"]].median().assign(classifier="best3").reset_index())
cal2 = (ev.assign(kind=np.where(ev.representation == "raw", "raw", "embedded")).groupby(["model", "kind"])
        [["cal_intercept", "cal_slope", "brier"]].median().reset_index().rename(columns={"model": "classifier"}))
pd.concat([cal, cal2]).to_csv(OUT / "calibration.csv", index=False)
in_ctrl = [p in CONTROLS for p in zip(ev.timepoint, ev.target)]
cal3 = (ev.assign(set=np.where(in_ctrl, "controls", "other pairs"), kind=np.where(ev.representation == "raw", "raw", "embedded"),
                  brier_prevalence=ev.prevalence * (1 - ev.prevalence),
                  logit_prevalence=np.log(ev.prevalence / (1 - ev.prevalence)))
        .groupby(["model", "set", "kind"])[["cal_intercept", "logit_prevalence", "cal_slope", "brier", "brier_prevalence"]]
        .median().reset_index())
cal3.to_csv(OUT / "calibration_by_set.csv", index=False)

vit = best[(best.target == "vitd") & (best.representation != "raw")]
vit_lr = lr[(lr.target == "vitd") & (lr.representation != "raw")]
counts = [dict(item="configurations evaluated", value=len(ev)),
          dict(item="pooled AUROC below 0.5, all classifiers", value=int((ev.auroc < .5).sum())),
          dict(item="AUROC below 0.5 with CI excluding 0.5", value=int((ev.auroc_hi < .5).sum())),
          dict(item="vitamin D embedded mean, best of 3", value=vit.auroc.mean()),
          dict(item="vitamin D embedded SD, best of 3", value=vit.auroc.std()),
          dict(item="vitamin D embedded mean, LR", value=vit_lr.auroc.mean()),
          dict(item="permutation null mean, minimum", value=pm.null_mean.min()),
          dict(item="permutation null mean, maximum", value=pm.null_mean.max()),
          dict(item="BH over all pairs, q below 0.05", value=int((pm.q_bh42 < .05).sum()))]
pd.DataFrame(counts).to_csv(OUT / "counts.csv", index=False)
print("wrote", OUT)
