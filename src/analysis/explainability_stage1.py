"""Stage 1 of the attribution analysis: attribution methods on the raw metabolome at 6 months.

Methods, each computed per outer fold and averaged:
  lr_coef    absolute elastic-net coefficient
  perm_lr    permutation importance of the LR (AUROC drop on the held-out fold)
  perm_rfc   permutation importance of a random forest
  univar_mi  mutual information per metabolite
Fixed hyperparameters, no inner tuning. Imputation and scaling are fitted on the training
rows. Stability is the mean pairwise Spearman correlation of the fold rankings and the
top-20 Jaccard index. For fish oil, the ranks of the known markers are reported.

Writes explainability/stage1_feature_ranks.csv, stage1_summary.csv and stage1_summary.md.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from metabo import config, data_io
from metabo.models import get_classifier

TP = "6m"
OUTCOMES = ["fishoil", "sex"]
NICE = {"fishoil": "Fish oil", "sex": "Sex"}
TOPK = 20
SEED = config.SEED
OUTDIR = config.RESULTS_DIR / "explainability"
OUTDIR.mkdir(exist_ok=True)


def load_sex(index):
    """Sex (male = 1) by ABCno from the source file, aligned to index."""
    d = pd.read_csv(config.SOURCE_FILE, sep=";", usecols=["ABCno", "GENDER"], low_memory=False)
    s = (d.dropna(subset=["GENDER"]).drop_duplicates("ABCno")
         .set_index("ABCno")["GENDER"].map({"Male": 1, "Female": 0}))
    return s.reindex(index)


def get_xy(outcome):
    X = data_io.load_metabolites(TP)
    if outcome == "sex":
        y = load_sex(X.index)
    else:
        y = data_io.load_targets()[outcome]
    return data_io.align(X, y)


def ground_truth_cols(columns):
    """Fish-oil markers: the CMPF family and the n3 fatty acids."""
    prim = [c for c in columns if "CMPF" in c or "furanpropano" in c
            or "furanpropionate" in c]
    n3 = [c for c in columns if any(t in c for t in ("20:5n3", "22:6n3", "22:5n3"))]
    return {"CMPF_primary": [c for c in columns if c.startswith(config.FISHOIL_MARKER_PARENT)],
            "CMPF_family": prim, "n3_pufa": n3, "all": sorted(set(prim) | set(n3))}


def attributions(Xtr, ytr, Xte, yte):
    """Importance of every metabolite under each method, for one fold."""
    out = {}
    lr = get_classifier("lr", random_state=SEED, n_jobs=1)
    lr.fit(Xtr, ytr)
    out["lr_coef"] = np.abs(lr.coef_.ravel())
    out["perm_lr"] = permutation_importance(
        lr, Xte, yte, scoring="roc_auc", n_repeats=10,
        random_state=SEED, n_jobs=-1).importances_mean

    rfc = get_classifier("rfc", random_state=SEED, n_jobs=-1).set_params(
        max_depth=5, n_estimators=400, min_samples_leaf=5)
    rfc.fit(Xtr, ytr)
    out["perm_rfc"] = permutation_importance(
        rfc, Xte, yte, scoring="roc_auc", n_repeats=5,
        random_state=SEED, n_jobs=-1).importances_mean

    out["univar_mi"] = mutual_info_classif(Xtr, ytr, random_state=SEED)
    return out


def ranks_from_importance(imp):
    """Rank 1 = most important (largest importance)."""
    order = np.argsort(-imp)
    r = np.empty_like(order)
    r[order] = np.arange(1, len(imp) + 1)
    return r


def run_outcome(outcome):
    X, y = get_xy(outcome)
    feats = list(X.columns)
    Xv, yv = X.values, y.values.astype(int)
    print(f"\n{NICE[outcome]} 6m: n={len(yv)} events={int(yv.sum())} p={Xv.shape[1]}")

    methods = ["lr_coef", "perm_lr", "perm_rfc", "univar_mi"]
    fold_imp = {m: [] for m in methods}
    outer = StratifiedKFold(config.OUTER_K, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(outer.split(Xv, yv)):
        pre = SimpleImputer(strategy="median")
        sc = RobustScaler()
        Xtr = sc.fit_transform(pre.fit_transform(Xv[tr]))
        Xte = sc.transform(pre.transform(Xv[te]))
        att = attributions(Xtr, yv[tr], Xte, yv[te])
        for m in methods:
            fold_imp[m].append(att[m])
        print(f"  fold {k+1}/{config.OUTER_K} done")

    rows, summary = [], []
    for m in methods:
        M = np.vstack(fold_imp[m])  # folds x metabolites
        fold_ranks = np.vstack([ranks_from_importance(M[i]) for i in range(M.shape[0])])
        mean_imp = M.mean(0)
        agg_rank = ranks_from_importance(mean_imp)
        sp = [spearmanr(fold_ranks[i], fold_ranks[j]).correlation
              for i, j in itertools.combinations(range(M.shape[0]), 2)]
        tops = [set(np.argsort(fold_ranks[i])[:TOPK]) for i in range(M.shape[0])]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
        for f, mi, ar in zip(feats, mean_imp, agg_rank):
            rows.append({"outcome": outcome, "method": m, "metabolite": f,
                         "mean_importance": float(mi), "rank": int(ar)})
        rec = {"outcome": outcome, "method": m,
               "stability_spearman": float(np.nanmean(sp)),
               "top20_jaccard": float(np.mean(jac))}
        if outcome == "fishoil":
            gt = ground_truth_cols(feats)
            rank_of = dict(zip(feats, agg_rank))
            def med_rank(cols):
                rs = [rank_of[c] for c in cols if c in rank_of]
                return float(np.median(rs)) if rs else np.nan
            rec["cmpf_rank"] = (med_rank(gt["CMPF_primary"]) if gt["CMPF_primary"]
                                else np.nan)
            rec["cmpf_family_med_rank"] = med_rank(gt["CMPF_family"])
            rec["n3_pufa_med_rank"] = med_rank(gt["n3_pufa"])
            rec["gt_all_med_rank"] = med_rank(gt["all"])
        summary.append(rec)
    return rows, summary, feats


def main():
    all_rows, all_sum = [], []
    for oc in OUTCOMES:
        r, s, _ = run_outcome(oc)
        all_rows += r
        all_sum += s

    ranks_df = pd.DataFrame(all_rows)
    sum_df = pd.DataFrame(all_sum)
    ranks_df.to_csv(OUTDIR / "stage1_feature_ranks.csv", index=False)
    sum_df.to_csv(OUTDIR / "stage1_summary.csv", index=False)

    lines = ["# Stage 1: attribution methods on the raw metabolome, 6 months\n",
             "Rank 1 = most important. Fish oil should rank the CMPF family and n3 fatty acids high.\n",
             "## Summary\n", sum_df.round(3).to_markdown(index=False), "\n",
             "## Top-10 metabolites per outcome x method\n"]
    for oc in OUTCOMES:
        lines.append(f"\n### {NICE[oc]}\n")
        for m in ranks_df["method"].unique():
            sub = (ranks_df[(ranks_df.outcome == oc) & (ranks_df.method == m)]
                   .sort_values("rank").head(10))
            names = [n.split("_")[0][:38] for n in sub["metabolite"]]
            lines.append(f"- **{m}**: " + "; ".join(names))
    (OUTDIR / "stage1_summary.md").write_text("\n".join(lines))
    print(f"\nWrote:\n  {OUTDIR/'stage1_feature_ranks.csv'}"
          f"\n  {OUTDIR/'stage1_summary.csv'}\n  {OUTDIR/'stage1_summary.md'}")
    print("\nfish-oil marker ranks")
    print(sum_df[sum_df.outcome == "fishoil"].round(2).to_string(index=False))
    print("\nstability over folds")
    print(sum_df[["outcome", "method", "stability_spearman", "top20_jaccard"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
