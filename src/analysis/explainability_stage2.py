"""Stage 2 of the attribution analysis: attribution through an embedding, mapped back to
the metabolites and compared with the raw lr_coef ranking from Stage 1.

Per outer fold the embedding and an elastic-net LR (fixed hyperparameters) are fitted on
the training rows, and the LR coefficients b over the k components are mapped back:
  pca   |V' b|, the exact metabolite weights of the linear model (checked against the
        classifier output in every fold)
  umap  |Pearson r| between each UMAP axis and each metabolite, times |b|
  vae   mean absolute decoder Jacobian, times |b| (only if torch is installed)
The importances are averaged over folds and ranked.
Reports embedded AUROC, Spearman and top-20 Jaccard against the raw ranking, and the
rank of hydroxy-CMPF.

Writes explainability/stage2_backprojection.csv and stage2_summary.md.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch and numpy ship separate OpenMP libraries on macOS
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # numba's OpenMP layer crashes next to torch on macOS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from metabo import config
from metabo.models import get_classifier
from metabo.embeddings import make_embedder
from explainability_stage1 import get_xy, NICE, OUTCOMES, TOPK

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
REPS = ["pca", "umap"] + (["vae"] if _HAS_TORCH else [])
DIMS = config.EMBED_DIMS
SEED = config.SEED
OUTDIR = config.RESULTS_DIR / "explainability"


def raw_reference_ranks(outcome):
    """Stage 1 lr_coef rank per metabolite."""
    r = pd.read_csv(OUTDIR / "stage1_feature_ranks.csv")
    sub = r[(r.outcome == outcome) & (r.method == "lr_coef")]
    return sub.set_index("metabolite")["rank"]


def _vae_sensitivity(embedder, Z, n_sample=128):
    """Mean absolute decoder Jacobian d x_hat / d z, (p, k), over the first n_sample
    training latents."""
    import torch
    from torch.func import jacrev, vmap
    Zt = torch.tensor(np.asarray(Z[:n_sample], dtype="float32"))
    with torch.no_grad():
        J = vmap(jacrev(lambda z: embedder.model_.dec(z)))(Zt)   # (n, p, k)
    return J.abs().mean(0).numpy()


def pca_weights(embedder, coef):
    """Metabolite weights w = V' b of an LR fitted on PCA scores. Exact: the model's
    decision function equals intercept + (x - mean) . w."""
    return embedder.components_.T @ coef


def check_pca_weights(embedder, clf, X):
    """Stop if the metabolite weights do not reproduce the classifier output."""
    w = pca_weights(embedder, clf.coef_.ravel())
    direct = clf.intercept_[0] + (X - embedder.mean_) @ w
    err = float(np.abs(direct - clf.decision_function(embedder.transform(X))).max())
    if err > 1e-6:
        raise RuntimeError(f"PCA weights do not reproduce the decision function (max error {err:.2e})")


def _magnitude_map(rep, embedder, Ztr, Xtr_scaled):
    """(p, k) non-negative sensitivity of each metabolite to each latent axis."""
    if rep == "pca":
        return np.abs(embedder.components_).T
    if rep == "vae":
        return _vae_sensitivity(embedder, Ztr)
    if rep == "umap":
        # UMAP has no inverse; use |Pearson r| between each axis and each metabolite
        Zc = Ztr - Ztr.mean(0)
        Xc = Xtr_scaled - Xtr_scaled.mean(0)
        zsd = Zc.std(0) + 1e-9
        xsd = Xc.std(0) + 1e-9
        return np.abs((Zc.T @ Xc) / (len(Ztr) * zsd[:, None] * xsd[None, :])).T
    raise ValueError(rep)


def backproject(rep, embedder, Ztr, coef, Xtr_scaled):
    """Per-metabolite importance from signed LR coefficients over the components.
    PCA uses the exact weights; UMAP and the VAE have no linear inverse, so a magnitude
    map is combined with |coef|."""
    if rep == "pca":
        return np.abs(pca_weights(embedder, coef))
    return _magnitude_map(rep, embedder, Ztr, Xtr_scaled) @ np.abs(coef)


def backproject_importance(rep, embedder, Ztr, imp, Xtr_scaled):
    """Per-metabolite importance from non-negative component importances (e.g. permutation
    importance). Negative permutation importances are noise and are set to zero. Without
    signs no exact mapping exists, so PCA also uses the magnitude map |V|'."""
    return _magnitude_map(rep, embedder, Ztr, Xtr_scaled) @ np.clip(imp, 0, None)


def ranks_from_importance(imp):
    order = np.argsort(-imp)
    r = np.empty(len(imp), dtype=int)
    r[order] = np.arange(1, len(imp) + 1)
    return r


def run(outcome):
    X, y = get_xy(outcome)
    feats = list(X.columns)
    Xv, yv = X.values, y.values.astype(int)
    ref = raw_reference_ranks(outcome).reindex(feats).values
    hcmpf_idx = [i for i, f in enumerate(feats) if config.FISHOIL_MARKER in f]
    print(f"\n{NICE[outcome]} 6m: n={len(yv)} events={int(yv.sum())}")

    rows = []
    for rep in REPS:
        for dim in DIMS:
            fold_imp, oof = [], np.full(len(yv), np.nan)
            outer = StratifiedKFold(config.OUTER_K, shuffle=True, random_state=SEED)
            for tr, te in outer.split(Xv, yv):
                pre, sc = SimpleImputer(strategy="median"), RobustScaler()
                Xtr = sc.fit_transform(pre.fit_transform(Xv[tr]))
                Xte = sc.transform(pre.transform(Xv[te]))
                emb = make_embedder(rep, dim, SEED)
                Ztr = emb.fit_transform(Xtr)
                Zte = emb.transform(Xte)
                clf = get_classifier("lr", random_state=SEED, n_jobs=1)
                clf.fit(Ztr, yv[tr])
                if rep == "pca":
                    check_pca_weights(emb, clf, Xte)
                fold_imp.append(backproject(rep, emb, Ztr, clf.coef_.ravel(), Xtr))
                oof[te] = clf.predict_proba(Zte)[:, 1]
            mean_imp = np.vstack(fold_imp).mean(0)
            bp_rank = ranks_from_importance(mean_imp)
            sp = spearmanr(bp_rank, ref).correlation
            top_bp = set(np.argsort(bp_rank)[:TOPK])
            top_raw = set(np.argsort(ref)[:TOPK])
            jac = len(top_bp & top_raw) / len(top_bp | top_raw)
            rec = {"outcome": outcome, "rep": rep, "dim": dim,
                   "embedded_auroc": float(roc_auc_score(yv, oof)),
                   "spearman_vs_raw": float(sp), "top20_jaccard_vs_raw": float(jac)}
            if hcmpf_idx:
                rec["hydroxyCMPF_backproj_rank"] = int(min(bp_rank[i] for i in hcmpf_idx))
                rec["hydroxyCMPF_raw_rank"] = int(min(ref[i] for i in hcmpf_idx))
            rows.append(rec)
            extra = (f" hCMPF {rec['hydroxyCMPF_backproj_rank']:>3}/{len(feats)}"
                     if hcmpf_idx else "")
            print(f"  {rep:4s} k={dim:<2d} AUROC={rec['embedded_auroc']:.3f} "
                  f"rho={sp:+.2f} J20={jac:.2f}{extra}")
    return rows


def main():
    rows = []
    for oc in OUTCOMES:
        rows += run(oc)
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "stage2_backprojection.csv", index=False)

    lines = ["# Stage 2: back-projected attribution against the raw lr_coef ranking, 6 months\n",
             df.round(3).to_markdown(index=False), "\n"]
    (OUTDIR / "stage2_summary.md").write_text("\n".join(lines))
    print(f"\nWrote {OUTDIR/'stage2_backprojection.csv'} and stage2_summary.md")
    fo = df[df.outcome == "fishoil"]
    print("\nfish oil, hydroxy-CMPF rank after back-projection")
    print(fo[["rep", "dim", "embedded_auroc", "spearman_vs_raw",
              "hydroxyCMPF_backproj_rank"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
