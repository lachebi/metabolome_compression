"""Model evaluation.

The pipeline is impute -> robust scaling -> [embedding] -> classifier. Scaling and the
embedding are fitted on the training rows of each fold. Nested CV: the outer
StratifiedKFold estimates performance, an inner StratifiedKFold tunes the classifier.
The embedding is fitted once per inner fold and reused for every hyperparameter combination.
Reported: AUROC and AUPRC with bootstrap 95% CIs on the pooled out-of-fold predictions,
calibration intercept, slope and Brier score.
The below-detection fill is done before this, in scripts/build_inputs.py.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, brier_score_loss)

from metabo import config
from metabo.models import get_classifier, PARAM_GRIDS
from metabo.embeddings import make_embedder


def build_embed_pipeline(representation, dim, random_state, n_neighbors=None):
    """impute -> scale -> [embed], without classifier; raw has no embed step."""
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", RobustScaler())]
    if representation != "raw":
        steps.append(("embed", make_embedder(representation, dim, random_state,
                                             n_neighbors=n_neighbors)))
    return Pipeline(steps)


def build_pipeline(representation, dim, model_name, random_state, n_neighbors=None,
                   clf_n_jobs=-1):
    """Full pipeline with default classifier settings, used by permutation_null."""
    pipe = build_embed_pipeline(representation, dim, random_state, n_neighbors)
    clf = get_classifier(model_name, random_state, n_jobs=clf_n_jobs)
    return Pipeline(pipe.steps + [("clf", clf)])


def _grid_combos(model_name):
    """Hyperparameter combinations as plain keyword arguments."""
    return [{k.replace("clf__", ""): v for k, v in combo.items()}
            for combo in ParameterGrid(PARAM_GRIDS[model_name])]


def _calibration(y, p):
    """Calibration intercept and slope from logistic recalibration on the logit."""
    eps = 1e-6
    p = np.clip(np.asarray(p), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        # Large C instead of no penalty, which warns on scikit-learn >= 1.8.
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit, y)
        return float(lr.intercept_[0]), float(lr.coef_[0, 0])
    except ValueError:
        # Expected when y is single-class (e.g. a degenerate bootstrap or permutation
        # draw); any other exception is a real bug and should propagate.
        return np.nan, np.nan


def _metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= 0.5).astype(int)
    two = len(np.unique(y)) > 1
    ci, slope = _calibration(y, p)
    return {
        "auroc": roc_auc_score(y, p) if two else np.nan,
        "auprc": average_precision_score(y, p) if two else np.nan,
        "mcc":   matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, p),
        "cal_intercept": ci,
        "cal_slope": slope,
    }


def _bootstrap_ci(y, p, fn, n_boot, random_state, alpha=0.05):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(fn(y[idx], p[idx]))
    if not stats:
        return np.nan, np.nan
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def nested_cv_oof(X, y, representation, dim, model_name, random_state=42,
                  outer_k=None, inner_k=None, n_neighbors=None, clf_n_jobs=-1):
    """Pooled out-of-fold probabilities from nested CV.

    The inner folds pick the hyperparameters with the highest summed AUROC; the winner is
    refitted, embedding included, on the outer training rows.
    """
    outer_k = outer_k or config.OUTER_K
    inner_k = inner_k or config.INNER_K
    Xv = X.values
    yv = np.asarray(y)
    oof = np.full(len(yv), np.nan)
    outer = StratifiedKFold(outer_k, shuffle=True, random_state=random_state)
    combos = _grid_combos(model_name)

    for tr, te in outer.split(Xv, yv):
        Xtr, ytr, Xte = Xv[tr], yv[tr], Xv[te]

        inner = StratifiedKFold(inner_k, shuffle=True, random_state=random_state)
        scores = np.zeros(len(combos))
        for itr, iva in inner.split(Xtr, ytr):
            emb = build_embed_pipeline(representation, dim, random_state, n_neighbors)
            Zitr = emb.fit_transform(Xtr[itr])
            Ziva = emb.transform(Xtr[iva])
            if len(np.unique(ytr[iva])) < 2:
                continue
            for ci, params in enumerate(combos):
                clf = get_classifier(model_name, random_state,
                                     n_jobs=clf_n_jobs).set_params(**params)
                clf.fit(Zitr, ytr[itr])
                scores[ci] += roc_auc_score(ytr[iva],
                                            clf.predict_proba(Ziva)[:, 1])
        best = combos[int(np.argmax(scores))]

        emb = build_embed_pipeline(representation, dim, random_state, n_neighbors)
        Ztr = emb.fit_transform(Xtr)
        Zte = emb.transform(Xte)
        clf = get_classifier(model_name, random_state,
                             n_jobs=clf_n_jobs).set_params(**best)
        clf.fit(Ztr, ytr)
        oof[te] = clf.predict_proba(Zte)[:, 1]
    return yv, oof


def evaluate(X, y, representation, dim, model_name, random_state=42,
             n_bootstrap=None, clf_n_jobs=-1):
    """Metrics for one representation, dimension and classifier.

    Returns (record, (y, out-of-fold probabilities)).
    """
    n_bootstrap = config.N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    yv, p = nested_cv_oof(X, y, representation, dim, model_name, random_state,
                          clf_n_jobs=clf_n_jobs)
    rec = _metrics(yv, p)
    for key, fn in [("auroc", roc_auc_score), ("auprc", average_precision_score)]:
        lo, hi = _bootstrap_ci(yv, p, fn, n_bootstrap, random_state)
        rec[f"{key}_lo"], rec[f"{key}_hi"] = lo, hi
    rec["n"] = int(len(yv))
    rec["n_events"] = int(yv.sum())
    rec["prevalence"] = float(yv.mean())
    return rec, (yv, p)


def permutation_null(X, y, representation, dim, model_name, n_perm=None,
                     random_state=42, cv_k=5, clf_n_jobs=-1):
    """Label-permutation test of the AUROC.

    One stratified CV with default hyperparameters (no inner tuning), on the same outer
    folds as nested_cv_oof, for the observed and every permuted label vector.
    p = (1 + #{null >= observed}) / (n_perm + 1).
    """
    n_perm = config.N_PERMUTATION if n_perm is None else n_perm
    rng = np.random.default_rng(random_state)
    Xv = X.values

    def cv_auroc(yy):
        skf = StratifiedKFold(cv_k, shuffle=True, random_state=random_state)
        oof = np.full(len(yy), np.nan)
        for tr, te in skf.split(Xv, yy):
            pipe = build_pipeline(representation, dim, model_name, random_state,
                                  clf_n_jobs=clf_n_jobs)
            pipe.fit(Xv[tr], yy[tr])
            oof[te] = pipe.predict_proba(Xv[te])[:, 1]
        return roc_auc_score(yy, oof)

    yv = np.asarray(y)
    obs = cv_auroc(yv)
    null = np.array([cv_auroc(rng.permutation(yv)) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return {
        "obs_auroc": float(obs),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float(p),
        "n_perm": int(n_perm),
    }
