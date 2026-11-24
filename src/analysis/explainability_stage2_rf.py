"""Stage 2 with a random forest instead of the LR.

The attribution over the k components is the RF permutation importance on the held-out
fold. Negative importances (noise) are set to zero and the rest are mapped back with
explainability_stage2.backproject_importance. Permutation importance has no sign, so the
PCA mapping here is the magnitude map |V|', not the exact weights used for the LR. The
reference is the raw perm_rfc ranking from Stage 1.

Writes explainability/stage2_rf_backprojection.csv and stage2_rf_summary.md.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # numba's OpenMP layer crashes next to torch on macOS
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from metabo import config
from metabo.models import get_classifier
from metabo.embeddings import make_embedder
from explainability_stage1 import get_xy, NICE, OUTCOMES, TOPK
from explainability_stage2 import backproject_importance, ranks_from_importance, REPS, DIMS

OUTDIR = config.RESULTS_DIR / "explainability"
SEED = config.SEED


def raw_rf_ranks(outcome):
    """Stage 1 perm_rfc rank per metabolite."""
    r = pd.read_csv(OUTDIR / "stage1_feature_ranks.csv")
    sub = r[(r.outcome == outcome) & (r.method == "perm_rfc")]
    return sub.set_index("metabolite")["rank"]


def rf():
    return get_classifier("rfc", random_state=SEED, n_jobs=-1).set_params(
        max_depth=5, n_estimators=400, min_samples_leaf=5)


def run(outcome):
    X, y = get_xy(outcome)
    feats = list(X.columns)
    Xv, yv = X.values, y.values.astype(int)
    ref = raw_rf_ranks(outcome).reindex(feats).values
    hcmpf_idx = [i for i, f in enumerate(feats) if config.FISHOIL_MARKER in f]
    print(f"\n{NICE[outcome]} 6m (RF downstream): n={len(yv)} events={int(yv.sum())}")
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
                Ztr, Zte = emb.fit_transform(Xtr), emb.transform(Xte)
                clf = rf().fit(Ztr, yv[tr])
                comp_imp = permutation_importance(
                    clf, Zte, yv[te], scoring="roc_auc", n_repeats=10,
                    random_state=SEED, n_jobs=-1).importances_mean
                fold_imp.append(backproject_importance(rep, emb, Ztr, comp_imp, Xtr))
                oof[te] = clf.predict_proba(Zte)[:, 1]
            bp_rank = ranks_from_importance(np.vstack(fold_imp).mean(0))
            sp = spearmanr(bp_rank, ref).correlation
            jac = (len(set(np.argsort(bp_rank)[:TOPK]) & set(np.argsort(ref)[:TOPK]))
                   / len(set(np.argsort(bp_rank)[:TOPK]) | set(np.argsort(ref)[:TOPK])))
            rec = {"outcome": outcome, "rep": rep, "dim": dim, "model": "rfc",
                   "embedded_auroc": float(roc_auc_score(yv, oof)),
                   "spearman_vs_raw": float(sp), "top20_jaccard_vs_raw": float(jac)}
            if hcmpf_idx:
                rec["hydroxyCMPF_backproj_rank"] = int(min(bp_rank[i] for i in hcmpf_idx))
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
    df.to_csv(OUTDIR / "stage2_rf_backprojection.csv", index=False)
    lines = ["# Stage 2 with a random forest: back-projected permutation importance against "
             "the raw perm_rfc ranking\n", df.round(3).to_markdown(index=False)]
    (OUTDIR / "stage2_rf_summary.md").write_text("\n".join(lines))
    fo = df[df.outcome == "fishoil"]
    print("\nfish oil, hydroxy-CMPF rank, random forest")
    print(fo[["rep", "dim", "embedded_auroc", "hydroxyCMPF_backproj_rank"]]
          .to_string(index=False))
    print(f"\nWrote {OUTDIR/'stage2_rf_backprojection.csv'} and stage2_rf_summary.md")


if __name__ == "__main__":
    main()
