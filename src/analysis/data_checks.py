"""Checks on the input data; no model is fitted. Writes results/stats/data_checks.csv
(item, value) and endpoint_correlations.csv: children per age, outcome correlations, shift
between ages, the fill rule, missingness, the size of a training-only fill, PC1 against
the number of filled values, and PCA variance.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from metabo import config, data_io

OUT = config.RESULTS_DIR / "stats"
OUT.mkdir(exist_ok=True)
TP_MAP = {"6m": "6m", "18mth": "18m", "6y": "6y"}
ROSTER = config.OUTCOMES
rows = []


def add(item, value):
    rows.append(dict(item=item, value=value))


X = {t: pd.read_csv(config.METAB_FILES[t], index_col=0) for t in config.TIMEPOINTS}
cols = list(X["6m"].columns)
add("input matrices, missing values", int(sum(x.isna().sum().sum() for x in X.values())))

# children per age and overlap
ids = {t: set(x.index) for t, x in X.items()}
for t in ids:
    add(f"children at {t}", len(ids[t]))
add("children, any age", len(set.union(*ids.values())))
add("children, all three ages", len(set.intersection(*ids.values())))

# outcome correlations
y = data_io.load_targets()[ROSTER].apply(pd.to_numeric, errors="coerce")
c = y.corr()
u = c.where(np.triu(np.ones(c.shape, bool), 1)).stack().rename("r").reset_index()
u = u.reindex(u.r.abs().sort_values(ascending=False).index)
u.to_csv(OUT / "endpoint_correlations.csv", index=False)
add("endpoint pairs", len(u))
add("endpoint pairs with abs r of 0.3 or more", int((u.r.abs() >= .3).sum()))
both = y[["sens_food", "sens_s_ig_e_any"]].dropna()
add("food-sensitized children", int(both.sens_food.sum()))
add("food-sensitized and specific IgE positive", int((both.sens_food * both.sens_s_ig_e_any).sum()))

# shift between ages in the same children (may include batch differences)
for a, b in [("6m", "18m"), ("18m", "6y"), ("6m", "6y")]:
    i = X[a].index.intersection(X[b].index)
    A, B = X[a].loc[i], X[b].loc[i]
    pooled = np.sqrt((A.values.var(0, ddof=1) + B.values.var(0, ddof=1)) / 2)
    add(f"age shift {a}-{b}, median abs standardized mean difference (pooled SD)",
        float(np.median(np.abs(B.values.mean(0) - A.values.mean(0)) / pooled)))
    add(f"age shift {a}-{b}, median same-child r",
        float(np.nanmedian([np.corrcoef(A.values[:, j], B.values[:, j])[0, 1] for j in range(len(cols))])))

# fill rule, checked against the source
src = pd.read_csv(config.SOURCE_FILE, sep=";", decimal=",", low_memory=False)
add("source samples", len(src))
all_met = src[cols].apply(pd.to_numeric, errors="coerce")
ch = src[src.TPn.isin(TP_MAP)]
add("child samples", len(ch))
met = all_met.loc[ch.index]
na = met.isna().values
add("filled values", int(na.sum()))
add("metabolites with filled values", int(na.any(0).sum()))
for tt, t in TP_MAP.items():
    add(f"filled share at {t}", float(na[ch.TPn.values == tt].mean()))
add("filled share per child, minimum", float(na.mean(1).min()))
add("filled share per child, maximum", float(na.mean(1).max()))
add("filled share per metabolite across ages, maximum", float(na.mean(0).max()))
for tt, t in TP_MAP.items():
    add(f"filled share per metabolite at {t}, maximum", float(na[ch.TPn.values == tt].mean(0).max()))

global_fill = np.log10(met.min(axis=0).values / 10)
L = pd.concat([X[TP_MAP[tt]].loc[ch.ABCno[ch.TPn == tt].values][cols] for tt in TP_MAP]).values
order = np.concatenate([np.where(ch.TPn.values == tt)[0] for tt in TP_MAP])
na_o = na[order]
match_all = np.isclose(L[na_o], np.broadcast_to(global_fill, L.shape)[na_o])
add("filled values matching the across-ages minimum/10", int(match_all.sum()))
if not match_all.all():
    raise RuntimeError("filled values in the input matrices do not equal the across-ages minimum/10")
perage = np.vstack([np.broadcast_to(np.log10(met[ch.TPn.values == tt].min(axis=0).values / 10),
                                    ((ch.TPn == tt).sum(), len(cols))) for tt in TP_MAP])
add("filled values that would match a per-age minimum/10", int(np.isclose(L[na_o], perage[na_o]).sum()))

# missingness
add("metabolites in analysis", len(cols))
add("maximum missing share, all samples", float(all_met.isna().mean().max()))
add("metabolites over 1/3 missing, child samples", int((met.isna().mean() > 1 / 3).sum()))
add("maximum missing share, child samples", float(met.isna().mean().max()))
ncols = [k for k in src.columns if k not in ("ABCno", "TPn", "PARENT_ID") and "_" in k]
nm = src[ncols].apply(pd.to_numeric, errors="coerce")
nm = nm.loc[:, nm.notna().any()]
add("source metabolite columns", nm.shape[1])
add("source columns at or below 1/3 missing, all samples", int((nm.isna().mean() <= 1 / 3).sum()))

# how much a training-only minimum would change, over the outer folds of the positive controls
targets = data_io.load_targets()
long_ids = ch.ABCno.values
ratios, changed, ratios_f, changed_f, dropped = [], [], [], [], []
has_fill = na.any(0)
for t, tar in [("6m", "sex"), ("6m", "fishoil"), ("18m", "sex"), ("6y", "sex")]:
    Xa, ya = data_io.align(X[t], targets[tar])
    for tr, _ in StratifiedKFold(config.OUTER_K, shuffle=True, random_state=config.SEED).split(Xa.values, ya):
        keep = np.isin(long_ids, Xa.index.values[tr])
        m = met.values[keep]
        dropped.append(int((np.isnan(m).mean(axis=0) > 1 / 3).sum()))
        fold_min = np.nanmin(m, axis=0)
        all_min = met.min(axis=0).values
        r = fold_min / all_min
        changed.append(float((r > 1 + 1e-12).mean()))
        ratios.append(r[r > 1 + 1e-12])
        rf = r[has_fill]
        changed_f.append(float((rf > 1 + 1e-12).mean()))
        ratios_f.append(rf[rf > 1 + 1e-12])
ratios, ratios_f = np.concatenate(ratios), np.concatenate(ratios_f)
add("training-only filter: metabolites dropped per fold, minimum", min(dropped))
add("training-only filter: metabolites dropped per fold, maximum", max(dropped))
add("training-only, metabolites with fills: share with a new minimum, per fold", float(np.mean(changed_f)))
add("training-only, metabolites with fills: median ratio where changed", float(np.median(ratios_f)))
add("training-only, metabolites with fills: maximum ratio", float(ratios_f.max()))
add("training-only: share of metabolites with a new minimum, per fold", float(np.mean(changed)))
add("training-only: median ratio where changed", float(np.median(ratios)))
add("training-only: 95th percentile ratio", float(np.percentile(ratios, 95)))
add("training-only: maximum ratio", float(ratios.max()))
add("training-only: median shift in log10 units", float(np.median(np.log10(ratios))))

# PC1 against the number of filled values per child, and PCA variance
for tt, t in TP_MAP.items():
    Xs = RobustScaler().fit_transform(X[t])
    pc = PCA(1, random_state=config.SEED).fit_transform(Xs)[:, 0]
    if not (ch.ABCno[ch.TPn == tt].values == X[t].index.values).all():
        raise RuntimeError(f"child order in the source and the {t} matrix differ")
    n = met[ch.TPn.values == tt].isna().sum(axis=1).values
    add(f"PC1 vs filled values, abs r, {t}", float(abs(np.corrcoef(pc, n)[0, 1])))
    p = PCA(random_state=config.SEED).fit(Xs)
    add(f"variance kept by 16 components, {t}", float(p.explained_variance_ratio_[:16].sum()))
    add(f"components for 90% variance, {t}", int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), .9) + 1))

d = pd.DataFrame(rows)
d.to_csv(OUT / "data_checks.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 80)
print(d.round(3).to_string(index=False))
print(u.head(6).round(2).to_string(index=False))
