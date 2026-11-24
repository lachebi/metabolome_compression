"""Figure 3 of the paper (figures/fig6_explainability_compression.png), from
explainability/stage2_fold_seeds_summary.csv (written by summarize_fold_seeds.py): medians over
the fold seeds (ten in the paper), VAE by the encoder gradient.

Panel A: median back-projected rank of hydroxy-CMPF for fish oil by k (rank 1 on the raw
metabolome).
Panel B: Spearman correlation of the back-projected ranking with the raw ranking, mean and
SD over k of the per-k medians, for sex and fish oil.
Panel C: share of the raw AUROC margin above chance kept by the embedded model (median)
against that correlation (median), one point per outcome, embedding and k.
"""
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metabo import config

NICE = {"fishoil": "Fish oil", "sex": "Sex"}
COL = {"pca": "#2c7fb8", "umap": "#d95f0e", "vae": "#31a354"}


def raw_auroc():
    """Nested-CV LR AUROC on the raw metabolome at 6 months."""
    recs = [r for f in glob.glob(str(config.RESULTS_DIR / "eval_*.json")) for r in json.load(open(f))]
    return {r["target"]: r["auroc"] for r in recs if r["timepoint"] == "6m" and r["representation"] == "raw"
            and r["model"] == "lr" and r["target"] in NICE}


RAW_AUROC = raw_auroc()
df = pd.read_csv(config.RESULTS_DIR / "explainability" / "stage2_fold_seeds_summary.csv")
df = df.rename(columns={"rank_median": "hydroxyCMPF_backproj_rank", "spearman_median": "spearman_vs_raw",
                        "auroc_median": "embedded_auroc"})
assert df.n_runs.nunique() == 1 and df.n_runs.iloc[0] >= 1, "every row must summarize the same number of fold seeds"
df = df[df.outcome.isin(NICE)]
NP = pd.read_csv(config.METAB_FILES["6m"], index_col=0, nrows=0).shape[1]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

REPS = [r for r in ("pca", "umap", "vae") if r in set(df.rep)]

fo = df[df.outcome == "fishoil"]
for rep in REPS:
    s = fo[fo.rep == rep].sort_values("dim")
    ax[0].plot(s["dim"], s["hydroxyCMPF_backproj_rank"], "-o", color=COL[rep],
               label=rep.upper())
ax[0].axhline(1, color="k", ls="--", lw=1)
ax[0].text(9, 45, "raw rank = 1 (dashed)", ha="center", va="bottom", fontsize=9)
ax[0].axhline((NP + 1) / 2, color="grey", ls=":", lw=1)
ax[0].text(16, (NP + 1) / 2 - 8, "expected rank under a random ordering", ha="right", va="bottom", fontsize=8, color="grey")
ax[0].set_ylim(NP + 10, -25)
ax[0].set_xlabel("embedding dimension k")
ax[0].set_ylabel("hydroxy-CMPF back-projected rank (median)")
ax[0].set_title("A", loc="left", fontweight="bold")
ax[0].legend(frameon=False)

x = np.arange(len(NICE))
w = 0.8 / len(REPS)
for j, rep in enumerate(REPS):
    means = [df[(df.outcome == oc) & (df.rep == rep)]["spearman_vs_raw"].mean()
             for oc in NICE]
    errs = [df[(df.outcome == oc) & (df.rep == rep)]["spearman_vs_raw"].std()
            for oc in NICE]
    ax[1].bar(x + (j - (len(REPS) - 1) / 2) * w, means, w, yerr=errs, capsize=3,
              color=COL[rep], label=rep.upper())
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks(x)
ax[1].set_xticklabels([NICE[o] for o in NICE], rotation=15, ha="right")
ax[1].set_ylabel("Spearman vs raw attribution\n(median per k; mean, SD over k)")
ax[1].set_title("B", loc="left", fontweight="bold")
ax[1].legend(frameon=False)

for rep in REPS:
    s = df[df.rep == rep]
    raw = s["outcome"].map(RAW_AUROC).values
    acc = (s["embedded_auroc"].values - 0.5) / (raw - 0.5)
    ax[2].scatter(acc, s["spearman_vs_raw"], color=COL[rep], alpha=0.75,
                  s=40, label=rep.upper())
ax[2].axhline(0, color="k", lw=0.8)
ax[2].axvline(0, color="grey", ls=":", lw=1)
ax[2].text(0.0, ax[2].get_ylim()[1], " chance", fontsize=8, color="grey", va="top")
ax[2].set_xlabel("share of the raw margin above chance kept\n(embedded AUROC - 0.5) / (raw AUROC - 0.5)")
ax[2].set_ylabel("Spearman concordance vs raw attribution")
ax[2].set_title("C", loc="left", fontweight="bold")
ax[2].legend(frameon=False)

fig.tight_layout()
out = config.RESULTS_DIR / "figures" / "fig6_explainability_compression.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
