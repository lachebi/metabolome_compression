"""Summary tables from the result files.

RQ1: permutation p per pair, with BH q-values within each timepoint.
RQ2: mean AUROC by representation and k over the pairs with p < 0.05.
Writes results/summary_rq1_predictiveness.csv, summary_auroc_by_target_dim.csv and
summary_auroc_vs_dim.csv.
"""
import glob
import json

import numpy as np
import pandas as pd

from metabo import config
from metabo.stats import bh as _bh


def _load(kind):
    rows = []
    for f in glob.glob(str(config.RESULTS_DIR / f"{kind}_*.json")):
        rows += json.load(open(f))
    df = pd.DataFrame(rows)
    if not df.empty and "target" in df.columns:
        df = df[df["target"].isin(config.OUTCOMES)].reset_index(drop=True)
    return df


def _dedupe_perm(pm):
    """Keep the record with the most permutations per pair; fail on true duplicates."""
    if pm.empty or "key" not in pm.columns or "n_perm" not in pm.columns:
        return pm
    top = pm[pm["n_perm"] == pm.groupby("key")["n_perm"].transform("max")]
    if top["key"].duplicated().any():
        dups = list(top.loc[top["key"].duplicated(keep=False), "key"].unique()[:5])
        raise SystemExit(f"duplicate permutation records at the same n_perm for "
                         f"{dups}; clean the results folder before summarizing.")
    dropped = len(pm) - len(top)
    if dropped:
        print(f"dropped {dropped} permutation records with fewer permutations")
    return top.reset_index(drop=True)


def main():
    ev = _load("eval")
    pm = _dedupe_perm(_load("permutation"))

    if pm.empty:
        print("no permutation results")
    else:
        # BH within each timepoint (14 outcomes), NOT the manuscript's BH over all 42 pairs
        # (that one is pair_statistics.py's q_bh42 / build_table_s1.py's raw_perm_q_bh42).
        pm["q_value"] = pm.groupby("timepoint")["p_value"].transform(
            lambda s: _bh(s.values))
        pm["predictive"] = pm["p_value"] < 0.05
        pm["fdr_sig"] = pm["q_value"] < 0.05
        print("RQ1, permutation test on the raw metabolome")
        print(f"{int(pm.predictive.sum())}/{len(pm)} pairs at p < 0.05; "
              f"{int(pm.fdr_sig.sum())}/{len(pm)} at q < 0.05 within timepoint")
        cols = ["timepoint", "target", "n", "n_events", "obs_auroc",
                "p_value", "q_value", "predictive", "fdr_sig"]
        print(pm.sort_values("p_value")[cols].to_string(index=False))
        out_rq1 = config.RESULTS_DIR / "summary_rq1_predictiveness.csv"
        pm.sort_values(["timepoint", "p_value"])[
            ["timepoint", "target", "n", "n_events", "obs_auroc", "null_mean",
             "p_value", "q_value", "predictive", "fdr_sig"]].round(4).to_csv(
            out_rq1, index=False)
        print(f"wrote {out_rq1}")
        predictive = set(map(tuple, pm.loc[pm.predictive, ["timepoint", "target"]].values))
    if ev.empty:
        print("no eval results")
        return
    if pm.empty:
        predictive = set(map(tuple, ev[["timepoint", "target"]].drop_duplicates().values))

    ev["pred_subset"] = ev.apply(lambda r: (r.timepoint, r.target) in predictive, axis=1)

    # Best of the three classifiers per cell, chosen on the outer estimate, so these
    # values are optimistic for raw and embeddings alike. Every k is reported.
    bydim = (ev.sort_values("auroc", ascending=False)
               .groupby(["timepoint", "target", "representation", "dim"]).head(1))
    sub = bydim[bydim["pred_subset"]]

    print("\nRQ2, mean AUROC by k over the pairs with p < 0.05")
    curve = sub.pivot_table(index="dim", columns="representation",
                            values="auroc", aggfunc="mean")
    raw_ref = sub.loc[sub.representation == "raw", "auroc"].mean()
    print(f"raw: {raw_ref:.3f}")
    print(curve.round(3).to_string())

    print("\nembedding minus raw")
    for rep in ["pca", "umap", "vae"]:
        if rep not in curve.columns:
            continue
        for dim, val in curve[rep].dropna().items():
            d = val - raw_ref
            print(f"{rep:4s} dim={dim:<2}: AUROC={val:.3f}  D_vs_raw={d:+.3f}")

    out_full = config.RESULTS_DIR / "summary_auroc_by_target_dim.csv"
    bydim[["timepoint", "target", "representation", "dim", "model", "pred_subset",
           "auroc", "auroc_lo", "auroc_hi", "auprc", "mcc",
           "cal_slope", "n", "n_events"]].round(4).to_csv(out_full, index=False)
    out_curve = config.RESULTS_DIR / "summary_auroc_vs_dim.csv"
    curve.round(4).to_csv(out_curve)
    print(f"\nwrote {out_full}\nwrote {out_curve}")


if __name__ == "__main__":
    main()
