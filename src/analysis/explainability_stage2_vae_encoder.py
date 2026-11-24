"""VAE attribution through the encoder.

The prediction path is x -> robust scaling -> encoder mean mu(x) -> eta = b0 + b' mu(x_s).
The gradient g_ij = d eta / d x_s,j = sum_l b_l d mu_l(x_i) / d x_s,j is computed per
training row; importance per fold is the mean |g_ij|, then the mean over folds, then the
rank. The encoder is nonlinear, so this is a local summary, not an exact map. The decoder
Jacobian of explainability_stage2.py is computed on the same fits for comparison.
Writes explainability/stage2_vae_encoder.csv.

--seeds S [S ...]  repeat with other seeds; --vary vae changes only the VAE seed,
                   --vary both also the folds. Writes stage2_vae_encoder_seeds.csv.
--null N           fish oil at k = 8 and 16 with N shuffled label vectors (seeds 1000 on),
                   folds and VAE seed varied. Writes stage2_vae_encoder_null.csv.
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # numba's OpenMP layer crashes next to torch on macOS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from torch.func import jacrev, vmap

from explainability_stage1 import TOPK, get_xy
from metabo import config
from metabo.embeddings import make_embedder
from metabo.models import get_classifier

SEED = config.SEED
# The null (shuffled-label) comparison is reported at k = 8 and 16 only, to keep its 100
# reruns per k feasible; intersect with config.EMBED_DIMS so a reduced EMBED_DIMS (e.g. for
# a quick synthetic-data run) narrows this instead of crashing a later summary stage.
NULL_DIMS = [d for d in (8, 16) if d in config.EMBED_DIMS]
OUTCOMES = ["fishoil", "sex"]
OUTDIR = config.RESULTS_DIR / "explainability"


def ranks(v):
    r = np.empty(len(v), dtype=int)
    r[np.argsort(-v)] = np.arange(1, len(v) + 1)
    return r


def encoder_gradients(model, x, b, device):
    xt = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float32, device=device)
    eta = model.encode(xt)[0] @ bt
    # Rows do not interact, so the gradient of the sum holds each row's own gradient.
    grad, = torch.autograd.grad(eta.sum(), xt)
    return grad.cpu().numpy(), eta.detach().cpu().numpy()


def run(outcomes, dims, vae_seed=SEED, fold_seed=SEED, shuffle_seed=None):
    rows = []
    stage1 = pd.read_csv(OUTDIR / "stage1_feature_ranks.csv")
    for outcome in outcomes:
        X, y = get_xy(outcome)
        Xv, yv, feats = X.values, np.asarray(y), list(X.columns)
        if shuffle_seed is not None:
            yv = np.random.default_rng(shuffle_seed).permutation(yv)
        ref = (stage1[(stage1.outcome == outcome) & (stage1.method == "lr_coef")]
               .set_index("metabolite")["rank"].reindex(feats).values)
        hcmpf = [i for i, f in enumerate(feats) if config.FISHOIL_MARKER in f]
        for dim in dims:
            enc, enc_signed, dec, oof, checks = [], [], [], np.full(len(yv), np.nan), []
            for tr, te in StratifiedKFold(config.OUTER_K, shuffle=True, random_state=fold_seed).split(Xv, yv):
                pre, sc = SimpleImputer(strategy="median"), RobustScaler()
                Xtr = sc.fit_transform(pre.fit_transform(Xv[tr]))
                Xte = sc.transform(pre.transform(Xv[te]))
                emb = make_embedder("vae", dim, vae_seed)
                Ztr = emb.fit_transform(Xtr)
                Zte = emb.transform(Xte)
                clf = get_classifier("lr", random_state=SEED, n_jobs=1).fit(Ztr, yv[tr])
                b = clf.coef_.ravel()
                oof[te] = clf.predict_proba(Zte)[:, 1]
                model = emb.model_.eval()
                g, eta = encoder_gradients(model, Xtr, b, emb.device_)
                enc.append(np.abs(g).mean(0))
                enc_signed.append(np.abs(g.mean(0)))
                # the gradient must reproduce the predictor change for a small step
                step = np.random.default_rng(SEED).normal(0, 1e-3, Xtr.shape).astype("float32")
                _, eta2 = encoder_gradients(model, Xtr + step, b, emb.device_)
                denom = np.abs(eta2 - eta).mean() + 1e-12
                checks.append(float(np.abs((eta2 - eta) - (g * step).sum(1)).mean() / denom))
                if checks[-1] > 0.05:
                    raise RuntimeError(f"encoder gradient does not reproduce a finite step (relative error {checks[-1]:.3f})")
                # decoder Jacobian, as in explainability_stage2.py
                Zt = torch.tensor(Ztr[:128].astype("float32"), device=emb.device_)
                with torch.no_grad():
                    J = vmap(jacrev(lambda z: model.dec(z)))(Zt)
                dec.append(J.abs().mean(0).cpu().numpy() @ np.abs(b))
            for name, folds in [("encoder_gradient", enc), ("encoder_abs_mean_gradient", enc_signed), ("decoder_jacobian", dec)]:
                rank = ranks(np.vstack(folds).mean(0))
                top, top_ref = set(np.argsort(rank)[:TOPK]), set(np.argsort(ref)[:TOPK])
                rows.append(dict(outcome=outcome, dim=dim, variant=name, embedded_auroc=roc_auc_score(yv, oof),
                                 spearman_vs_raw=spearmanr(rank, ref).correlation,
                                 top20_jaccard_vs_raw=len(top & top_ref) / len(top | top_ref),
                                 gradient_check=np.mean(checks),
                                 hydroxyCMPF_rank=min(rank[i] for i in hcmpf)))
            print(outcome, dim, f"vae_seed={vae_seed} fold_seed={fold_seed} done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--vary", choices=["vae", "both"], default="vae")
    ap.add_argument("--null", type=int, default=0)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.null:
        if not NULL_DIMS:
            raise SystemExit(f"--null needs k = 8 or 16 in config.EMBED_DIMS ({config.EMBED_DIMS})")
        path = OUTDIR / "stage2_vae_encoder_null.csv"
        done = pd.read_csv(path) if path.exists() else pd.DataFrame()
        for seed in range(1000, 1000 + a.null):
            if len(done) and (done.seed == seed).any():
                continue
            d = run(["fishoil"], NULL_DIMS, seed, seed, shuffle_seed=seed)
            d.insert(0, "seed", seed)
            done = pd.concat([done, d])
            done.to_csv(path, index=False)
    elif not a.seeds:
        d = run(OUTCOMES, config.EMBED_DIMS)
        d.to_csv(OUTDIR / "stage2_vae_encoder.csv", index=False)
        print(d.round(3).to_string(index=False))
    else:
        path = OUTDIR / "stage2_vae_encoder_seeds.csv"
        done = pd.read_csv(path) if path.exists() else pd.DataFrame()
        for seed in a.seeds:
            if len(done) and ((done.seed == seed) & (done.vary == a.vary)).any():
                continue
            d = run(OUTCOMES, config.EMBED_DIMS, seed, seed if a.vary == "both" else SEED)
            d.insert(0, "seed", seed)
            d.insert(1, "vary", a.vary)
            done = pd.concat([done, d])
            done.to_csv(path, index=False)
