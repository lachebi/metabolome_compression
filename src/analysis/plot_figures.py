"""Figures and the per-pair table for RQ1, from the permutation and eval results.

  figures/fig4_significance.png        AUROC against -log10 permutation p (paper Figure 1A)
  figures/fig2_rq1_predictiveness.png  nested-CV AUROC with 95% CI for pairs above 0.6 (Figure 1B)
  figures/fig1_auroc_vs_dim.png        mean AUROC by k, pairs with p < 0.05
  figures/fig3_predictive_per_timepoint.png  pairs with p < 0.05 per timepoint
  results_rq1_complete.csv             both AUROC estimates, CI, p and q per pair
"""
import glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metabo import config
from metabo.stats import bh as _bh

FIGDIR = config.RESULTS_DIR / "figures"
FIGDIR.mkdir(exist_ok=True)

NICE = {
    "AD 6 yr ever": "AD ever",
    "Asthma 6 yr ever": "Asthma ever",
    "ar_6y": "Allergic rhinitis",
    "atopy_temp": "Atopy",
    "sens_food": "Food sens.",
    "sens_inhal": "Inhalant sens.",
    "sens_s_ig_e_any": "Spec. IgE",
    "fetal-Antibiotics during birth": "Antibiotics at birth",
    "fetal-Antibiotics during pregnancy": "Antibiotics in pregnancy",
    "fetal-Delivery": "Delivery mode",
    "mother_smoking": "Maternal smoking",
    "fishoil": "Fish oil",
    "vitd": "Vitamin D",
    "sex": "Sex",
}


def nice(t):
    return NICE.get(t, t)


def load_perm():
    """Permutation records (most permutations per pair) with BH q within timepoint."""
    recs = []
    for f in glob.glob(str(config.RESULTS_DIR / "permutation_*.json")):
        recs += json.load(open(f))
    recs = [r for r in recs if r["target"] in config.OUTCOMES]
    best = {}
    for r in recs:
        k = r["key"]
        if k not in best or r["n_perm"] > best[k]["n_perm"]:
            best[k] = r
    df = pd.DataFrame(best.values())
    # BH within each timepoint (14 outcomes), NOT the manuscript's BH over all 42 pairs
    # (that one is pair_statistics.py's q_bh42 / build_table_s1.py's raw_perm_q_bh42).
    df["q_value"] = np.nan
    for tp, idx in df.groupby("timepoint").groups.items():
        df.loc[idx, "q_value"] = _bh(df.loc[idx, "p_value"].values)
    return df


def fig1_auroc_vs_dim():
    df = pd.read_csv(config.RESULTS_DIR / "summary_auroc_vs_dim.csv").set_index("dim")
    raw = float(df.loc[0, "raw"]) if 0 in df.index and not pd.isna(df.loc[0, "raw"]) else np.nan
    dims = [d for d in df.index if d != 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for rep, color in [("pca", "C0"), ("umap", "C1"), ("vae", "C2")]:
        if rep in df.columns:
            ax.plot(dims, df.loc[dims, rep], "o-", color=color, label=rep.upper())
    if not np.isnan(raw):
        ax.axhline(raw, ls="--", color="k", label=f"raw = {raw:.3f}")
    ax.axhline(0.5, color="pink", label="chance")
    ax.set_xlabel("embedding dimension"); ax.set_ylabel("mean AUROC (predictive subset)")
    ax.set_title("Mean AUROC by k, pairs with p < 0.05")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_auroc_vs_dim.png", dpi=150); plt.close(fig)


def load_eval_raw_lr():
    """Nested-CV LR records on the raw metabolome, one per pair."""
    recs = []
    for f in glob.glob(str(config.RESULTS_DIR / "eval_*.json")):
        recs += json.load(open(f))
    df = pd.DataFrame([r for r in recs
                       if r.get("representation") == "raw" and r.get("model") == "lr"
                       and r.get("target") in config.OUTCOMES])
    cols = ["timepoint", "target", "auroc", "auroc_lo", "auroc_hi",
            "auprc", "mcc", "brier", "prevalence", "n", "n_events"]
    return df[[c for c in cols if c in df.columns]]


def complete_table(perm, ev):
    """One row per pair: both AUROC estimates, CI, permutation p and q."""
    p = perm[["timepoint", "target", "obs_auroc", "null_mean", "p_value",
              "q_value", "n_perm"]].rename(columns={"obs_auroc": "perm_auroc",
                                                     "null_mean": "perm_null_mean"})
    e = ev.rename(columns={"auroc": "eval_auroc", "auroc_lo": "eval_auroc_lo",
                           "auroc_hi": "eval_auroc_hi"})
    d = e.merge(p, on=["timepoint", "target"], how="outer")
    d.insert(1, "outcome", d["target"].map(nice))
    d["both_above_0.5"] = (d["eval_auroc"] > 0.5) & (d["perm_auroc"] > 0.5)
    d["both_above_0.6"] = (d["eval_auroc"] > 0.6) & (d["perm_auroc"] > 0.6)
    d["ci_excludes_0.5"] = d["eval_auroc_lo"] > 0.5
    d["fdr_tier"] = pd.cut(d["q_value"], [-0.01, 0.05, 0.10, 1.01],
                           labels=["q<0.05", "0.05-0.10", "ns"])
    d["significant_p05"] = d["p_value"] < 0.05
    return d.sort_values(["timepoint", "p_value", "eval_auroc"],
                         ascending=[True, True, False])


def _tier_color(p):
    return "#08519c" if p < 0.05 else "#cccccc"


def fig2_rq1(tab, thr=0.6):
    """Pairs with both AUROC estimates above thr: nested-CV AUROC with 95% CI, colored by
    permutation p < 0.05, and the permutation-test AUROC as a cross."""
    d = tab[(tab["eval_auroc"] > thr) & (tab["perm_auroc"] > thr)].copy()
    d["label"] = d["target"].map(nice) + " – " + d["timepoint"]
    d = d.sort_values("eval_auroc", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, max(5, 0.34 * len(d))))
    for y, r in d.iterrows():
        col = _tier_color(r["p_value"])
        ax.errorbar(r["eval_auroc"], y,
                    xerr=[[r["eval_auroc"] - r["eval_auroc_lo"]],
                          [r["eval_auroc_hi"] - r["eval_auroc"]]],
                    fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.4, capsize=3)
        ax.plot(r["perm_auroc"], y, marker="x", ms=5, color="#555555")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["label"], fontsize=7)
    ax.set_ylim(-1, len(d))
    ax.set_xlim(min(0.5, d["eval_auroc_lo"].min() - 0.02),
                min(1.0, d["eval_auroc_hi"].max() + 0.03))
    ax.axvline(0.5, ls="--", lw=1, color="red")
    ax.axvline(0.6, ls="--", lw=1, color="green")
    ax.set_xlabel("AUROC (raw metabolome, LR)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color="#08519c", label="p < 0.05"),
               Line2D([0], [0], marker="o", ls="", color="#cccccc", label="ns (p ≥ 0.05)"),
               Line2D([0], [0], marker="x", ls="", color="#555555",
                      label="permutation AUROC (single CV)"),
               Line2D([0], [0], ls="--", color="red", label="AUROC = 0.5"),
               Line2D([0], [0], ls="--", color="green", label="AUROC = 0.6")]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    ax.set_title("B", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_rq1_predictiveness.png", dpi=150); plt.close(fig)


def fig4_significance(df):
    """Permutation-test AUROC against -log10(p), per pair."""
    d = df.copy()
    d["neglogp"] = -np.log10(d["p_value"].clip(lower=1e-4))
    def color(r):
        if r["p_value"] < 0.05: return "#08519c"
        return "#bbbbbb"
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(d["obs_auroc"], d["neglogp"], c=d.apply(color, axis=1), s=28,
               edgecolor="white", linewidth=0.4)
    ax.axhline(-np.log10(0.05), ls="--", lw=1, color="C3", label="p = 0.05")
    ax.axvline(0.5, ls="--", lw=1, color="red", label="AUROC = 0.5 (chance)")
    ax.axvline(0.6, ls="--", lw=1, color="green", label="AUROC = 0.6")
    # labels for p < 0.05 in a column at the right, with leader lines
    sig = d[d["p_value"] < 0.05].sort_values("neglogp", ascending=False).reset_index(drop=True)
    xmax = d["obs_auroc"].max()
    ax.set_xlim(0.34, xmax + 0.30)
    label_x = xmax + 0.06
    ymax = d["neglogp"].max()
    label_ys = np.linspace(ymax, 0.05, len(sig))
    for i, r in sig.iterrows():
        ax.annotate(f"{nice(r['target'])} – {r['timepoint']}",
                    xy=(r["obs_auroc"], r["neglogp"]),
                    xytext=(label_x, label_ys[i]), fontsize=7, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999",
                                    shrinkA=0, shrinkB=2))
    ax.set_xlabel("observed AUROC (raw, single CV)")
    ax.set_ylabel(r"$-\log_{10}$ (permutation p-value)")
    ax.set_title("A", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_significance.png", dpi=150); plt.close(fig)


def fig3_per_tp(df):
    order = [tp for tp in config.TIMEPOINTS if tp in set(df["timepoint"])]
    counts = {tp: ((df.timepoint == tp) & (df.p_value < 0.05)).sum() for tp in order}
    totals = {tp: (df.timepoint == tp).sum() for tp in order}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(order, [counts[tp] for tp in order], color="C0")
    for tp, b in zip(order, bars):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{counts[tp]}/{totals[tp]}", ha="center", va="bottom")
    ax.set_ylabel("predictive outcomes (p<0.05)")
    ax.set_title("Pairs with p < 0.05 per timepoint")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_predictive_per_timepoint.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    perm = load_perm()
    tab = complete_table(perm, load_eval_raw_lr())
    csv_path = config.RESULTS_DIR / "results_rq1_complete.csv"
    tab.to_csv(csv_path, index=False, float_format="%.4f")
    fig1_auroc_vs_dim()
    fig2_rq1(tab)
    fig3_per_tp(perm)
    fig4_significance(perm)
    print(f"wrote 4 figures to {FIGDIR} and {csv_path} ({len(tab)} pairs)")
    perm = perm.copy(); perm["outcome"] = perm["target"].map(nice)
    show = perm.sort_values(["timepoint", "p_value"])[
        ["timepoint", "outcome", "n", "n_events", "obs_auroc", "p_value"]]
    sig = show[show["p_value"] < 0.05]
    print(f"{len(sig)}/{len(show)} pairs at p < 0.05")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(sig.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    per_tp = show.assign(sig=show["p_value"] < 0.05).groupby("timepoint")["sig"].sum()
    print("per timepoint:", per_tp.to_dict())
