"""PCA and UMAP attribution over several fold seeds, as for the VAE encoder runs
(explainability_stage2_vae_encoder.py --vary both).

The seed sets the outer folds and the UMAP random_state (PCA is deterministic given the
folds). PCA uses the exact signed mapping, |w| per fold, mean over folds, then rank.
UMAP uses the correlation surrogate of explainability_stage2.py.

--seeds S [S ...]  seeds (default 42 to 51); writes explainability/stage2_fold_seeds.csv
--null N           fish oil at k = 8 and 16 with N shuffled label vectors (seeds 1000 on);
                   writes stage2_fold_seeds_null.csv
Seed 42 reproduces stage2_pca_signed.csv for PCA.
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # numba's OpenMP layer crashes next to torch on macOS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from explainability_stage1 import OUTCOMES, TOPK, get_xy
from metabo import config
from metabo.embeddings import make_embedder
from metabo.models import get_classifier

SEED = config.SEED
OUTDIR = config.RESULTS_DIR / "explainability"
# The null (shuffled-label) comparison is reported at k = 8 and 16 only, to keep its 100
# reruns per k feasible; intersect with config.EMBED_DIMS so a reduced EMBED_DIMS (e.g. for
# a quick synthetic-data run) narrows this instead of crashing a later summary stage.
NULL_DIMS = [d for d in (8, 16) if d in config.EMBED_DIMS]


def umap_surrogate(Ztr, Xtr, comp_imp):
    """explainability_stage2.backproject for UMAP. Not imported from there, because that
    module imports torch, which can crash next to UMAP's numba threads on macOS."""
    Zc, Xc = Ztr - Ztr.mean(0), Xtr - Xtr.mean(0)
    zsd, xsd = Zc.std(0) + 1e-9, Xc.std(0) + 1e-9
    corr = (Zc.T @ Xc) / (len(Ztr) * zsd[:, None] * xsd[None, :])
    return np.abs(corr).T @ np.abs(comp_imp)


def ranks(v):
    r = np.empty(len(v), dtype=int)
    r[np.argsort(-v)] = np.arange(1, len(v) + 1)
    return r


def run(outcomes, dims, seed, shuffle_seed=None):
    stage1 = pd.read_csv(OUTDIR / "stage1_feature_ranks.csv")
    rows = []
    for outcome in outcomes:
        X, y = get_xy(outcome)
        Xv, yv, feats = X.values, np.asarray(y).astype(int), list(X.columns)
        if shuffle_seed is not None:
            yv = np.random.default_rng(shuffle_seed).permutation(yv)
        ref = (stage1[(stage1.outcome == outcome) & (stage1.method == "lr_coef")]
               .set_index("metabolite")["rank"].reindex(feats).values)
        hcmpf = [i for i, f in enumerate(feats) if config.FISHOIL_MARKER in f]
        for rep in ["pca", "umap"]:
            for dim in dims:
                imps, oof = [], np.full(len(yv), np.nan)
                for tr, te in StratifiedKFold(config.OUTER_K, shuffle=True, random_state=seed).split(Xv, yv):
                    pre, sc = SimpleImputer(strategy="median"), RobustScaler()
                    Xtr = sc.fit_transform(pre.fit_transform(Xv[tr]))
                    Xte = sc.transform(pre.transform(Xv[te]))
                    emb = make_embedder(rep, dim, seed)
                    Ztr = emb.fit_transform(Xtr)
                    Zte = emb.transform(Xte)
                    clf = get_classifier("lr", random_state=SEED, n_jobs=1).fit(Ztr, yv[tr])
                    b = clf.coef_.ravel()
                    if rep == "pca":
                        w = emb.components_.T @ b
                        err = np.abs(clf.intercept_[0] + (Xte - emb.mean_) @ w - clf.decision_function(Zte)).max()
                        if err > 1e-6:
                            raise RuntimeError(f"PCA weights do not reproduce the decision function ({err:.2e})")
                        imps.append(np.abs(w))
                    else:
                        imps.append(umap_surrogate(Ztr, Xtr, b))
                    oof[te] = clf.predict_proba(Zte)[:, 1]
                rank = ranks(np.vstack(imps).mean(0))
                top, top_ref = set(np.argsort(rank)[:TOPK]), set(np.argsort(ref)[:TOPK])
                rows.append(dict(outcome=outcome, rep=rep, dim=dim,
                                 embedded_auroc=roc_auc_score(yv, oof),
                                 spearman_vs_raw=spearmanr(rank, ref).correlation,
                                 top20_jaccard_vs_raw=len(top & top_ref) / len(top | top_ref),
                                 hydroxyCMPF_rank=min(rank[i] for i in hcmpf) if hcmpf else np.nan))
        print(outcome, f"seed={seed}", "shuffled" if shuffle_seed is not None else "", "done", flush=True)
    return pd.DataFrame(rows)


def append(path, d):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, d], ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--null", type=int, default=0)
    a = ap.parse_args()
    outcomes = OUTCOMES
    if a.null:
        if not NULL_DIMS:
            raise SystemExit(f"--null needs k = 8 or 16 in config.EMBED_DIMS ({config.EMBED_DIMS})")
        path = OUTDIR / "stage2_fold_seeds_null.csv"
        done = set(pd.read_csv(path).seed) if path.exists() else set()
        for seed in range(1000, 1000 + a.null):
            if seed in done:
                continue
            d = run(["fishoil"], NULL_DIMS, seed, shuffle_seed=seed)
            d.insert(0, "seed", seed)
            append(path, d)
    else:
        path = OUTDIR / "stage2_fold_seeds.csv"
        done = set(pd.read_csv(path).seed) if path.exists() else set()
        for seed in a.seeds:
            if seed in done:
                continue
            d = run(outcomes, config.EMBED_DIMS, seed)
            d.insert(0, "seed", seed)
            append(path, d)
