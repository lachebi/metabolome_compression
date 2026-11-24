"""Summary of the VAE encoder seed runs and the shuffled-label runs. Checks that seed 42
with the main folds reproduces stage2_vae_encoder.csv. Writes
explainability/stage2_vae_encoder_seed_summary.csv and stage2_vae_encoder_null_summary.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from metabo import config

O = config.RESULTS_DIR / "explainability"
s = pd.read_csv(O / "stage2_vae_encoder_seeds.csv")
main = pd.read_csv(O / "stage2_vae_encoder.csv")
k = ["outcome", "dim", "variant"]
c = main.merge(s[(s.seed == 42) & (s.vary == "vae")], on=k, suffixes=("", "_s"))
assert len(c) == len(main), (len(c), len(main))
assert (c.hydroxyCMPF_rank == c.hydroxyCMPF_rank_s).all(), "seed 42 does not reproduce the main file"
assert np.allclose(c.embedded_auroc, c.embedded_auroc_s), "seed 42 does not reproduce the main file"
g = s.groupby(["vary", "outcome", "variant", "dim"])
d = g.agg(n_seeds=("seed", "nunique"), auroc_median=("embedded_auroc", "median"),
          auroc_min=("embedded_auroc", "min"), auroc_max=("embedded_auroc", "max"),
          spearman_min=("spearman_vs_raw", "min"), spearman_max=("spearman_vs_raw", "max"),
          jaccard_max=("top20_jaccard_vs_raw", "max"),
          rank_median=("hydroxyCMPF_rank", "median"), rank_min=("hydroxyCMPF_rank", "min"),
          rank_max=("hydroxyCMPF_rank", "max"),
          n_rank_top20=("hydroxyCMPF_rank", lambda r: int((r <= 20).sum()))).reset_index()
d.to_csv(O / "stage2_vae_encoder_seed_summary.csv", index=False)
pd.set_option("display.width", 220)
print(d[(d.outcome == "fishoil") & (d.variant == "encoder_gradient")].round(3).to_string(index=False))
print(d[(d.outcome == "sex") & (d.variant == "encoder_gradient")].round(3).to_string(index=False))

null_path = O / "stage2_vae_encoder_null.csv"
if null_path.exists():
    n = pd.read_csv(null_path)
    n = n[n.variant == "encoder_gradient"]
    real = s[(s.outcome == "fishoil") & (s.variant == "encoder_gradient") & (s.vary == "both")]
    rows = []
    for label, e in [("true labels", real), ("shuffled labels", n)]:
        for dim, g in e[e.dim.isin([8, 16])].groupby("dim"):
            rows.append(dict(labels=label, dim=dim, n_runs=len(g), rank_median=g.hydroxyCMPF_rank.median(),
                             rank_min=g.hydroxyCMPF_rank.min(), rank_max=g.hydroxyCMPF_rank.max(),
                             n_top20=int((g.hydroxyCMPF_rank <= 20).sum()),
                             n_top100=int((g.hydroxyCMPF_rank <= 100).sum()),
                             auroc_median=g.embedded_auroc.median()))
            if label == "shuffled labels":
                t = real[real.dim == dim]
                u = g.hydroxyCMPF_rank
                rows[-1]["mwu_p_vs_true"] = mannwhitneyu(t.hydroxyCMPF_rank, u, alternative="less").pvalue
                rows[-1]["fisher_p_top20_vs_true"] = fisher_exact(
                    [[(t.hydroxyCMPF_rank <= 20).sum(), (t.hydroxyCMPF_rank > 20).sum()],
                     [(u <= 20).sum(), (u > 20).sum()]], alternative="greater")[1]
                rows[-1]["spearman_median"] = g.spearman_vs_raw.median()
                rows[-1]["jaccard_max"] = g.top20_jaccard_vs_raw.max()
            else:
                rows[-1]["spearman_median"] = g.spearman_vs_raw.median()
                rows[-1]["jaccard_max"] = g.top20_jaccard_vs_raw.max()
    r = pd.DataFrame(rows)
    r.to_csv(O / "stage2_vae_encoder_null_summary.csv", index=False)
    print(r.round(3).to_string(index=False))
