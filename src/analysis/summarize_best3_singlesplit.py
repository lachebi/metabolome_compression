"""Best-of-three-classifiers AUROC (original single split, nested cross-validation), by set
of outcome-timepoint pairs, representation and latent dimension k.

Reads supplement_S1_auroc_all_pairs.csv (already one row per timepoint x outcome x
representation x dim, with auroc_best_of_3 = the best of lr/rfc/knn on that cell's nested-CV
estimate) and aggregates it over three sets of pairs: the four positive controls (sex at
three timepoints, fish oil at 6 months), all 42 outcome-timepoint pairs, and the 11 pairs with
raw-metabolome permutation p < 0.05 (nominal_p05 column).

Known-answer check: Results 3.3 and the manuscript's own build/inventory.py already quote, from
this exact computation, "positive controls 0.713 against 0.51 to 0.54" and "all 42 pairs 0.569
against 0.52 to 0.54" -- raw mean of auroc_best_of_3, and the min-max range of the
(dim, representation) mean grid for the embeddings (not a mean of per-representation means: the
quoted range spans every dim x representation cell, matching how this quantity was first
computed). Reproducing those four numbers here, to the two decimals already in the manuscript
text, is the check.

Writes results/summary_best3_singlesplit.csv.
"""
import pandas as pd

from metabo import config

R = config.RESULTS_DIR
s1 = pd.read_csv(R / "supplement_S1_auroc_all_pairs.csv")

raw = s1[s1.representation == "raw"]
SETS = [
    ("positive controls", "Positive controls (4)",
     [("Sex", "6m"), ("Sex", "18m"), ("Sex", "6y"), ("Fish oil", "6m")]),
    ("all pairs", "All pairs (42)", None),
    ("nominally significant", "Nominally significant (11)", None),
]


def pairs_for(key, idx_list):
    if idx_list is not None:
        return idx_list
    if key == "all pairs":
        return list(zip(raw.outcome, raw.timepoint))
    if key == "nominally significant":
        sig = raw[raw.nominal_p05]
        return list(zip(sig.outcome, sig.timepoint))
    raise ValueError(key)


rows = []
for key, label, idx_list in SETS:
    idx = set(pairs_for(key, idx_list))
    sub = s1[[(o, t) in idx for o, t in zip(s1.outcome, s1.timepoint)]]
    n_pairs = len(idx)
    raw_mean = sub[sub.representation == "raw"].auroc_best_of_3.mean()
    for dim in config.EMBED_DIMS:
        for rep in ["pca", "umap", "vae"]:
            cell = sub[(sub.representation == rep) & (sub.dim == dim)]
            assert len(cell) == n_pairs, (key, rep, dim, len(cell), n_pairs)
            rows.append(dict(set=key, label=label, n_pairs=n_pairs, raw_mean=raw_mean,
                              dim=dim, representation=rep, auroc_best_of_3_mean=cell.auroc_best_of_3.mean()))

out = pd.DataFrame(rows).round(4)
out.to_csv(R / "summary_best3_singlesplit.csv", index=False)

# known-answer check
checks = {
    "positive controls": (0.713, 0.51, 0.54),
    "all pairs": (0.569, 0.52, 0.54),
}
print("set                      raw    emb range (over dim x rep)")
for key, (exp_raw, exp_lo, exp_hi) in checks.items():
    sub = out[out.set == key]
    got_raw = round(sub.raw_mean.iloc[0], 3)
    got_lo, got_hi = sub.auroc_best_of_3_mean.min(), sub.auroc_best_of_3_mean.max()
    print(f"{key:25s}{got_raw:.3f}  {got_lo:.3f} to {got_hi:.3f}")
    assert round(got_raw, 3) == exp_raw, (key, got_raw, exp_raw)
    assert round(got_lo, 2) == exp_lo, (key, got_lo, exp_lo)
    assert round(got_hi, 2) == exp_hi, (key, got_hi, exp_hi)
print("known-answer check passed: reproduces the manuscript's own quoted 0.713/0.51-0.54 and "
      "0.569/0.52-0.54 to two decimals")
print(f"\nwrote {R / 'summary_best3_singlesplit.csv'}, {len(out)} rows")
