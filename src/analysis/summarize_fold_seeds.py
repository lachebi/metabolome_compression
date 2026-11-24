"""Summary of the attribution seed runs for PCA, UMAP and the VAE encoder (same fold seeds),
and comparison of the hydroxy-CMPF rank with true and shuffled labels (one-sided
Mann-Whitney). Writes explainability/stage2_fold_seeds_summary.csv and
stage2_fold_seeds_null_summary.csv.
"""
import pandas as pd
from scipy.stats import mannwhitneyu

from metabo import config

O = config.RESULTS_DIR / "explainability"
s = pd.read_csv(O / "stage2_fold_seeds.csv")
n = pd.read_csv(O / "stage2_fold_seeds_null.csv")
v = pd.read_csv(O / "stage2_vae_encoder_seeds.csv")
v = v[(v.vary == "both") & (v.variant == "encoder_gradient")].assign(rep="vae")
vn = pd.read_csv(O / "stage2_vae_encoder_null.csv")
vn = vn[vn.variant == "encoder_gradient"].assign(rep="vae")
print(f"{s.seed.nunique()} seeds, {n.seed.nunique()} shuffled-label runs")
a = pd.concat([s, v[s.columns]], ignore_index=True)
nn = pd.concat([n, vn[n.columns]], ignore_index=True)

t = a.groupby(["outcome", "rep", "dim"]).agg(
    n_runs=("seed", "nunique"), auroc_median=("embedded_auroc", "median"),
    auroc_min=("embedded_auroc", "min"), auroc_max=("embedded_auroc", "max"),
    spearman_median=("spearman_vs_raw", "median"), spearman_min=("spearman_vs_raw", "min"),
    spearman_max=("spearman_vs_raw", "max"), jaccard_max=("top20_jaccard_vs_raw", "max"),
    rank_median=("hydroxyCMPF_rank", "median"), rank_min=("hydroxyCMPF_rank", "min"),
    rank_max=("hydroxyCMPF_rank", "max"),
    n_rank_top20=("hydroxyCMPF_rank", lambda r: int((r <= 20).sum()))).reset_index()
t.to_csv(O / "stage2_fold_seeds_summary.csv", index=False)

rows = []
for rep in ["pca", "umap", "vae"]:
    for dim in [8, 16]:
        tr = a[(a.outcome == "fishoil") & (a.rep == rep) & (a.dim == dim)].hydroxyCMPF_rank
        sh = nn[(nn.rep == rep) & (nn.dim == dim)].hydroxyCMPF_rank
        rows.append(dict(rep=rep, dim=dim, true_median=tr.median(), true_min=tr.min(), true_max=tr.max(),
                         true_top20=int((tr <= 20).sum()), shuffled_median=sh.median(), shuffled_min=sh.min(),
                         shuffled_max=sh.max(), shuffled_top20=int((sh <= 20).sum()), n_shuffled=len(sh),
                         mwu_p_true_better=mannwhitneyu(tr, sh, alternative="less").pvalue))
r = pd.DataFrame(rows)
r.to_csv(O / "stage2_fold_seeds_null_summary.csv", index=False)
pd.set_option("display.width", 220)
print(t.round(3).to_string(index=False))
print(r.round(4).to_string(index=False))
