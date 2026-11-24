"""Top attributed metabolites (absolute elastic-net coefficient) for sex and fish oil at
6 months on the raw metabolome, from explainability/stage1_feature_ranks.csv.
Writes figures/fig7_attribution_controls.png.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metabo import config

NICE = {"sex": "Sex", "fishoil": "Fish oil"}
COL = {"sex": "#6a51a3", "fishoil": "#2c7fb8"}
N_TOP = 10


def _short_name(metabolite):
    return metabolite.split("_")[0].strip()


def main():
    df = pd.read_csv(config.RESULTS_DIR / "explainability" / "stage1_feature_ranks.csv")
    lr = df[df.method == "lr_coef"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, outcome in zip(axes, ["sex", "fishoil"]):
        sub = lr[lr.outcome == outcome].sort_values("rank").head(N_TOP)
        names = [_short_name(m) for m in sub["metabolite"]][::-1]
        vals = sub["mean_importance"].values[::-1]
        ax.barh(range(len(names)), vals, color=COL[outcome], alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("|elastic-net coefficient|")
        ax.set_title(NICE[outcome], fontsize=12)
    n = lr.metabolite.nunique()
    fig.suptitle(f"Top {N_TOP} of {n} metabolites, raw metabolome, 6 months", fontsize=12)
    fig.tight_layout(rect=[0.01, 0, 0.99, 0.93])

    out = config.RESULTS_DIR / "figures" / "fig7_attribution_controls.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
