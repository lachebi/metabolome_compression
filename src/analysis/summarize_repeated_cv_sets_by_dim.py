"""Per-(set, representation, dim) breakdown of summarize_repeated_cv.py's "sets" table.

summary_sets.csv (summarize_repeated_cv.py) reports, for each set of outcome-timepoint pairs,
the raw-minus-embedding AUROC difference averaged over the five latent dimensions. This script
gives the same quantity broken out by dim, by aggregating summary_per_pair_setting.csv (one row
per pair, representation, dim) with the same "mean of the pair standard errors" convention used
for the sets table (not a fresh Nadeau-Bengio computation on dim-pooled data).

Known-answer check: averaging these per-dim rows over dim (plain mean of diff, plain mean of se,
matching how summary_sets.csv itself is built from per-pair rows) must reproduce summary_sets.csv's
"per pair" view for the same (set, representation), because both are unweighted means over the
same rectangular (pair x dim) grid, so the two orders of averaging commute for the mean. The two
paths do NOT have to agree on se/lo/hi to the same precision, because summary_sets.csv's se is a
mean of summary_per_pair.csv's per-pair se (itself from a fresh Nadeau-Bengio fit on dim-pooled
fold differences), not a mean of per-dim se; this is reported, not silently forced.

Writes results/repeated_cv/summary_sets_by_dim.csv.
"""
import pandas as pd

from metabo import config

R = config.RESULTS_DIR
OUT = R / "repeated_cv"
REPS = ["pca", "umap", "vae"]
DIMS = config.EMBED_DIMS

per_setting = pd.read_csv(OUT / "summary_per_pair_setting.csv")  # pair, representation, dim, diff, se, lo, hi
per_set = pd.read_csv(OUT / "summary_sets.csv")  # set, view, representation, n, diff, se, lo, hi (no dim)

# t-critical value implied by the existing per-dim rows (same for every row: same N_FOLDS/DFREE
# as summarize_repeated_cv.py); recovered from the data instead of recomputed from raw JSONs,
# since this script only aggregates already-computed per-pair-per-dim results.
t_implied = ((per_setting.hi - per_setting["diff"]) / per_setting.se)
T = t_implied.median()
assert (t_implied - T).abs().max() < 1e-6, "t-critical value is not constant across per-dim rows"

bh = pd.read_csv(R / "stats" / "bh_over_42.csv")
tp_col = "timepoint"
tg_col = "target" if "target" in bh.columns else "outcome"
nominal = set((bh[tp_col] + "|" + bh[tg_col])[bh["p_value"] < 0.05])
controls = {"6m|sex", "18m|sex", "6y|sex", "6m|fishoil"}
all_pairs = set(per_setting.pair.unique())
SETS_FOR_TABLE4 = {
    "positive controls": controls,
    "pairs with p < 0.05": nominal,
    "all pairs": all_pairs,
}

rows = []
for name, members in SETS_FOR_TABLE4.items():
    sub = per_setting[per_setting.pair.isin(members)]
    for rep in REPS:
        for dim in DIMS:
            g = sub[(sub.representation == rep) & (sub.dim == dim)]
            mean, se = g["diff"].mean(), g["se"].mean()
            rows.append(dict(set=name, representation=rep, dim=dim, n=len(g),
                              diff=mean, se=se, lo=mean - T * se, hi=mean + T * se))
out = pd.DataFrame(rows)
out.to_csv(OUT / "summary_sets_by_dim.csv", index=False)

print("known-answer check: mean over dim vs summary_sets.csv ('per pair' view)")
mismatches = []
for name in SETS_FOR_TABLE4:
    for rep in REPS:
        mine = out[(out.set == name) & (out.representation == rep)]
        mean_diff_over_k = mine["diff"].mean()
        mean_se_over_k = mine["se"].mean()
        ref = per_set.query("set == @name and view == 'per pair' and representation == @rep")
        assert len(ref) == 1, (name, rep)
        ref_diff, ref_se = ref["diff"].iloc[0], ref["se"].iloc[0]
        d_diff = abs(mean_diff_over_k - ref_diff)
        d_se = abs(mean_se_over_k - ref_se)
        print(f"  {name:22s} {rep:5s}  diff {mean_diff_over_k:+.4f} vs {ref_diff:+.4f} (|diff|={d_diff:.5f})"
              f"   se {mean_se_over_k:.4f} vs {ref_se:.4f} (|diff|={d_se:.5f})")
        assert d_diff < 0.0005, (name, rep, "diff does not reproduce summary_sets.csv", d_diff)
        if d_se > 0.0005:
            mismatches.append((name, rep, d_se))
if mismatches:
    print("\nNOTE: se differs by more than 0.0005 for the following (set, representation) pairs; "
          "diff matches exactly (unweighted means commute), se does not have to, because "
          "summary_sets.csv's se comes from a fresh Nadeau-Bengio fit on dim-pooled fold "
          "differences (summary_per_pair.csv), not from averaging the five per-dim se values:")
    for name, rep, d_se in mismatches:
        print(f"  {name} / {rep}: |se diff| = {d_se:.5f}")
print(f"\nWrote {OUT / 'summary_sets_by_dim.csv'} ({len(out)} rows)")
