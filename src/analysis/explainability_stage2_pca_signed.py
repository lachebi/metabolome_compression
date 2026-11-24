"""Exact PCA mapping of the logistic predictor to the metabolites.

For robust-scaled x_s, PCA centering c, loadings V (p x k) and coefficients b:
    eta = b0 + b' V' (x_s - c),  so in scaled units w = V b.
On the log10 scale w_log = w / IQR, with the IQRs of the training fold. w does not
change when a component flips sign, because b flips with it.

Variants written (rank of): the earlier |V|' |b| (old_abs_before_sum), |w| per fold
averaged over folds (signed_abs_then_mean, same order of operations as the raw
reference), |mean w| (signed_mean_then_abs), and |w_log| averaged over folds. The mapping
is checked against the model's decision function on the held-out rows.
Writes explainability/stage2_pca_signed.csv.
"""
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


def ranks(v):
    r = np.empty(len(v), dtype=int)
    r[np.argsort(-v)] = np.arange(1, len(v) + 1)
    return r


def score(rank, ref):
    top = set(np.argsort(rank)[:TOPK])
    top_ref = set(np.argsort(ref)[:TOPK])
    return spearmanr(rank, ref).correlation, len(top & top_ref) / len(top | top_ref)


rows = []
stage1 = pd.read_csv(OUTDIR / "stage1_feature_ranks.csv")
for outcome in OUTCOMES:
    X, y = get_xy(outcome)
    Xv, yv, feats = X.values, np.asarray(y), list(X.columns)
    ref = (stage1[(stage1.outcome == outcome) & (stage1.method == "lr_coef")]
           .set_index("metabolite")["rank"].reindex(feats).values)
    hcmpf = [i for i, f in enumerate(feats) if config.FISHOIL_MARKER in f]
    for dim in config.EMBED_DIMS:
        w_folds, wlog_folds, old_folds, max_err = [], [], [], 0.0
        oof = np.full(len(yv), np.nan)
        for tr, te in StratifiedKFold(config.OUTER_K, shuffle=True, random_state=SEED).split(Xv, yv):
            pre, sc = SimpleImputer(strategy="median"), RobustScaler()
            Xtr = sc.fit_transform(pre.fit_transform(Xv[tr]))
            Xte = sc.transform(pre.transform(Xv[te]))
            emb = make_embedder("pca", dim, SEED)
            Ztr = emb.fit_transform(Xtr)
            Zte = emb.transform(Xte)
            clf = get_classifier("lr", random_state=SEED, n_jobs=1).fit(Ztr, yv[tr])
            b = clf.coef_.ravel()
            V = emb.components_.T
            w = V @ b
            direct = clf.intercept_[0] + (Xte - emb.mean_) @ w
            err = float(np.abs(direct - clf.decision_function(Zte)).max())
            if err > 1e-6:
                raise RuntimeError(f"PCA weights do not reproduce the decision function ({err:.2e})")
            max_err = max(max_err, err)
            w_folds.append(w)
            wlog_folds.append(w / sc.scale_)
            old_folds.append(np.abs(V) @ np.abs(b))
            oof[te] = clf.predict_proba(Zte)[:, 1]
        W, WL = np.vstack(w_folds), np.vstack(wlog_folds)
        variants = {"old_abs_before_sum": np.vstack(old_folds).mean(0),
                    "signed_abs_then_mean": np.abs(W).mean(0),
                    "signed_mean_then_abs": np.abs(W.mean(0)),
                    "signed_log10_units": np.abs(WL).mean(0)}
        for name, imp in variants.items():
            rank = ranks(imp)
            rho, jac = score(rank, ref)
            rows.append(dict(outcome=outcome, dim=dim, variant=name,
                             embedded_auroc=roc_auc_score(yv, oof), spearman_vs_raw=rho,
                             top20_jaccard_vs_raw=jac, max_predictor_error=max_err,
                             hydroxyCMPF_rank=min(rank[i] for i in hcmpf) if hcmpf else np.nan))
d = pd.DataFrame(rows)
d.to_csv(OUTDIR / "stage2_pca_signed.csv", index=False)
pd.set_option("display.width", 200)
print(d.round(3).to_string(index=False))
