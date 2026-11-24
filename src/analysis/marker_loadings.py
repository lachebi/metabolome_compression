"""Share of each marker's variance kept by the first k principal components at 6 months:
sum over l <= k of V_jl^2 lambda_l / var_j. Writes explainability/marker_variance_kept.csv.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from metabo import config

X = pd.read_csv(config.METAB_FILES["6m"], index_col=0)
Xs = RobustScaler().fit_transform(X)
pca = PCA(random_state=config.SEED).fit(Xs)
var = Xs.var(axis=0, ddof=1)
cum = np.cumsum((pca.components_.T ** 2) * pca.explained_variance_, axis=1) / var[:, None]
markers = {"hydroxy-CMPF": "hydroxy-CMPF", "CMPF": "3-carboxy-4-methyl-5-propyl-2-furanpropanoate",
           "pregnanediol disulfate": "5alpha-pregnan-3beta,20alpha-diol disulfate",
           "pregnenetriol sulfate": "pregnenetriol sulfate", "androstenediol disulfate": "androstenediol"}
rows = []
KS = [k for k in [2, 3, 4, 8, 16, 50, 150] if k <= cum.shape[1]]
for k in KS:
    c = cum[:, k - 1]
    rows.append(dict(marker=f"median of {X.shape[1]}", k=k, share_kept=np.median(c), percentile=50.0))
    for label, pat in markers.items():
        idx = [i for i, f in enumerate(X.columns) if pat.lower() in f.lower()]
        for i in idx[:1]:
            rows.append(dict(marker=f"{label} [{X.columns[i][:40]}]", k=k, share_kept=c[i],
                             percentile=(c < c[i]).mean() * 100))
d = pd.DataFrame(rows).round(3)
d.to_csv(config.RESULTS_DIR / "explainability" / "marker_variance_kept.csv", index=False)
print(d.pivot(index="marker", columns="k", values="share_kept").to_string())
print(d[d.k == 16].to_string(index=False))
print("components with nonzero variance:", int((pca.explained_variance_ > 1e-10).sum()),
      "| variance by k:", {k: round(pca.explained_variance_ratio_[:k].sum(), 3) for k in [16, 50, 150, 300] if k <= len(pca.explained_variance_ratio_)})
