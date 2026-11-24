"""Summary of scripts/run_repeated_cv.py.

Differences are raw minus embedding, logistic regression, per fold, paired within a
repeat. Interval per pair: mean of the R x K fold differences with the Nadeau-Bengio
corrected variance (1/(R K) + 1/(K - 1)) x sample variance, t with R K - 1 df. Pairs share
children and outcomes, so the standard error of a set is the mean of the pair standard
errors (complete dependence). Writes results/repeated_cv/summary_*.csv.
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

from metabo import config

R = config.RESULTS_DIR
OUT = R / "repeated_cv"
REPS = ["pca", "umap", "vae"]

recs = []
for f in glob.glob(str(OUT / "repeated_*.json")):
    recs += json.load(open(f))
d = pd.DataFrame(recs).drop_duplicates("key")
d = d[d.target.isin(config.OUTCOMES)]
d["pair"] = d.timepoint + "|" + d.target
N_PAIRS = len(config.OUTCOMES) * len(config.TIMEPOINTS)
assert d.pair.nunique() == N_PAIRS, d.pair.nunique()
assert d.groupby("pair").size().nunique() == 1, "pairs have different numbers of records"
N_FOLDS = d.repeat.nunique() * config.OUTER_K
DFREE = N_FOLDS - 1
T = stats.t.ppf(0.975, DFREE)

# repeat 0 uses the seed of the main run
pub = []
for f in glob.glob(str(R / "eval_*.json")):
    pub += json.load(open(f))
pub = pd.DataFrame(pub)
pub = pub[(pub.model == "lr")].copy()
pub["pair"] = pub.timepoint + "|" + pub.target
m = d[d.repeat == 0].merge(pub, on=["pair", "representation", "dim"], suffixes=("", "_pub"))
assert len(m) == len(d[d.repeat == 0]), len(m)
check = m.assign(diff=(m.auroc - m.auroc_pub).abs()).groupby("representation")["diff"].agg(["median", "max"]).round(4)
print("repeat 0 against the main run, LR, absolute difference\n", check)
check.to_csv(OUT / "summary_repeat0_check.csv")
assert check.loc[["raw", "pca"], "max"].max() < 1e-6, "raw or PCA does not reproduce the main run"

# fold table: one row per pair, representation, k, repeat, fold
fold = d[["pair", "target", "timepoint", "representation", "dim", "repeat", "auroc", "fold_auroc", "n_events"]].explode("fold_auroc")
fold["fold"] = fold.groupby(["pair", "representation", "dim", "repeat"]).cumcount()
fold["fold_auroc"] = fold.fold_auroc.astype(float)
raw = fold[fold.representation == "raw"].set_index(["pair", "repeat", "fold"]).fold_auroc
emb = fold[fold.representation != "raw"].copy()
emb["diff"] = raw.reindex(pd.MultiIndex.from_frame(emb[["pair", "repeat", "fold"]])).values - emb.fold_auroc
assert emb["diff"].notna().all()


def nb(x):
    x = np.asarray(x, float)
    assert len(x) == N_FOLDS, len(x)
    return x.mean(), np.sqrt((1 / N_FOLDS + 1 / (config.OUTER_K - 1)) * x.var(ddof=1))


def row(mean, se):
    return dict(diff=mean, se=se, lo=mean - T * se, hi=mean + T * se)


rows = []
for (pair, rep, dim), g in emb.groupby(["pair", "representation", "dim"]):
    rows.append(dict(pair=pair, representation=rep, dim=dim, **row(*nb(g["diff"]))))
per_setting = pd.DataFrame(rows)
per_setting.to_csv(OUT / "summary_per_pair_setting.csv", index=False)

# average over k within repeat and fold, then the interval
avg = emb.groupby(["pair", "target", "representation", "repeat", "fold"])["diff"].mean().reset_index()
rows = []
for (pair, target, rep), g in avg.groupby(["pair", "target", "representation"]):
    rows.append(dict(pair=pair, target=target, representation=rep, **row(*nb(g["diff"]))))
per_pair = pd.DataFrame(rows)

# pooled out-of-fold values next to the fold values
pool = d.groupby(["pair", "representation", "dim", "repeat"]).auroc.first().reset_index()
pr = pool[pool.representation == "raw"].groupby("pair").auroc.mean().rename("raw_pooled")
pe = (pool[pool.representation != "raw"].groupby(["pair", "representation", "repeat"]).auroc.mean()
      .groupby(["pair", "representation"]).mean().rename("emb_pooled").reset_index())
per_pair = per_pair.merge(pe, on=["pair", "representation"]).merge(pr, on="pair")
per_pair["diff_pooled"] = per_pair.raw_pooled - per_pair.emb_pooled
events = d[d.representation == "raw"].groupby("pair").n_events.first()
per_pair["n_events"] = per_pair.pair.map(events)
per_pair.to_csv(OUT / "summary_per_pair.csv", index=False)

# sets
p = pd.read_csv(R / "stats" / "bh_over_42.csv")
pcol = "p" if "p" in p.columns else [c for c in p.columns if c.startswith("p")][0]
tp = "timepoint"; tg = "target" if "target" in p.columns else "outcome"
nominal = set((p[tp] + "|" + p[tg])[p[pcol] < 0.05])
controls = {"6m|sex", "18m|sex", "6y|sex", "6m|fishoil"}
sets = {"positive controls": controls, "pairs with p < 0.05": nominal,
        "other pairs": set(per_pair.pair) - nominal, "all pairs": set(per_pair.pair)}
rows = []
for name, members in sets.items():
    for rep in REPS:
        g = per_pair[(per_pair.pair.isin(members)) & (per_pair.representation == rep)]
        once = g.groupby("target")[["diff", "se"]].mean()
        for view, x in [("per pair", g), ("each outcome once", once)]:
            mean, se = x["diff"].mean(), x["se"].mean()
            rows.append(dict(set=name, view=view, representation=rep, n=len(x),
                             raw_pooled=g.raw_pooled.mean(), emb_pooled=g.emb_pooled.mean(),
                             **row(mean, se), excludes_zero=(mean - T * se) > 0 or (mean + T * se) < 0))
per_set = pd.DataFrame(rows)
per_set.to_csv(OUT / "summary_sets.csv", index=False)

# across outcomes: each outcome once (mean over timepoints of the per-pair difference, which is
# already averaged over k), two-sided Wilcoxon signed-rank test per embedding; descriptive,
# because some endpoints overlap
rows = []
for rep in REPS:
    per_outcome = per_pair[per_pair.representation == rep].groupby("target")["diff"].mean()
    assert len(per_outcome) == len(config.OUTCOMES), len(per_outcome)
    # the mean over outcomes must equal the "all pairs, each outcome once" set mean above
    set_mean = per_set.query("set == 'all pairs' and view == 'each outcome once' and representation == @rep")["diff"]
    assert len(set_mean) == 1 and abs(set_mean.iloc[0] - per_outcome.mean()) < 1e-12, (rep, set_mean)
    w = stats.wilcoxon(per_outcome.values, alternative="two-sided")
    rows.append(dict(representation=rep, n_outcomes=len(per_outcome), mean_diff=per_outcome.mean(),
                     n_favor_raw=int((per_outcome > 0).sum()), statistic=w.statistic, p=w.pvalue))
wil = pd.DataFrame(rows)
wil.to_csv(OUT / "summary_wilcoxon.csv", index=False)
print(wil.round(3).to_string(index=False))

pd.set_option("display.width", 220)
print(per_set.round(3).to_string(index=False))
print(per_pair.round(3).sort_values(["representation", "diff"], ascending=[True, False]).drop(columns="target").to_string(index=False))
