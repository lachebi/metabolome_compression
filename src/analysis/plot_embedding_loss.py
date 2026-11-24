"""Figure 2 of the paper (figures/fig5_embedding_signal_loss.png).

Panel A: fish oil at 6 months, best raw AUROC against the best AUROC of each embedding.
Panel B: best PCA against best VAE AUROC per pair, with the pairwise Pearson r of PCA,
UMAP and VAE.
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metabo import config

FIGDIR = config.RESULTS_DIR / "figures"
FIGDIR.mkdir(exist_ok=True)

def load_best():
    """Best AUROC over k and classifier per pair: raw[pair] and best[pair][rep]."""
    raw, best = {}, {}
    for f in glob.glob(str(config.RESULTS_DIR / "eval_*.json")):
        for r in json.load(open(f)):
            if r["target"] not in config.OUTCOMES:
                continue
            k = (r["timepoint"], r["target"]); a = r["auroc"]; rep = r["representation"]
            if rep == "raw":
                raw[k] = max(raw.get(k, 0), a)
            else:
                best.setdefault(k, {})
                best[k][rep] = max(best[k].get(rep, 0), a)
    return raw, best


def pearson(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    return float(np.corrcoef(xs, ys)[0, 1])


def main():
    raw, best = load_best()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    examples = [(("6m", "fishoil"), "Fish oil, 6 months")]
    reps = ["raw", "pca", "umap", "vae"]
    colors = ["#222222", "#08519c", "#d94801", "#6a51a3"]
    width = 0.18
    xbase = np.arange(len(examples))
    for j, rep in enumerate(reps):
        vals = []
        for (k, _) in examples:
            vals.append(raw[k] if rep == "raw" else best[k][rep])
        axA.bar(xbase + (j - 1.5) * width, vals, width,
                label=rep.upper() if rep != "raw" else "raw", color=colors[j])
    axA.axhline(0.5, ls="--", lw=1, color="red", label="chance")
    axA.set_xticks(xbase); axA.set_xticklabels([lab for _, lab in examples], fontsize=9)
    axA.set_ylim(0.45, 0.82); axA.set_ylabel("AUROC (best classifier, best dim)")
    axA.set_xlim(-0.5, 0.5)
    axA.set_title("A", loc="left", fontweight="bold")
    axA.legend(fontsize=8, ncol=2)

    keys = [k for k in best if all(rr in best[k] for rr in ("pca", "umap", "vae"))]
    pca = [best[k]["pca"] for k in keys]
    umap = [best[k]["umap"] for k in keys]
    vae = [best[k]["vae"] for k in keys]
    axB.scatter(pca, vae, s=22, color="#08519c", edgecolor="white", linewidth=0.4)
    lo, hi = 0.40, 0.75
    axB.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#888888")
    axB.set_xlim(lo, hi); axB.set_ylim(lo, hi)
    axB.set_xlabel("best PCA AUROC"); axB.set_ylabel("best VAE AUROC")
    r_pv = pearson(pca, vae); r_pu = pearson(pca, umap); r_uv = pearson(umap, vae)
    axB.set_title("B", loc="left", fontweight="bold")
    axB.text(0.97, 0.04, f"Pearson r: PCA-VAE {r_pv:.2f}\nPCA-UMAP {r_pu:.2f}, UMAP-VAE {r_uv:.2f}",
             transform=axB.transAxes, ha="right", va="bottom", fontsize=8)

    fig.tight_layout()
    out = FIGDIR / "fig5_embedding_signal_loss.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)
    print(f"  example: fishoil-6m raw={raw[('6m','fishoil')]:.3f} "
          f"bestEmb={max(best[('6m','fishoil')].values()):.3f}")
    print(f"  mirror r: PCA-VAE {r_pv:.3f}  PCA-UMAP {r_pu:.3f}  UMAP-VAE {r_uv:.3f} (n={len(keys)})")


if __name__ == "__main__":
    main()
