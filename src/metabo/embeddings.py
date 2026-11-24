"""PCA, UMAP and VAE as scikit-learn transformers, so they are fitted inside each fold.

UMAP is kept at fixed settings (config) and the VAE is tuned only on reconstruction
loss; neither sees the outcome. Set USE_CUML=1 to run UMAP with RAPIDS cuML on a GPU.
"""
import os

from sklearn.decomposition import PCA

from metabo import config


def make_embedder(mode, dim, random_state=42, n_neighbors=None, min_dist=None):
    if mode == "pca":
        return PCA(n_components=dim, random_state=random_state)

    if mode == "umap":
        n_neighbors = config.UMAP_N_NEIGHBORS if n_neighbors is None else n_neighbors
        min_dist = config.UMAP_MIN_DIST if min_dist is None else min_dist
        if os.environ.get("USE_CUML"):
            from cuml.manifold import UMAP as cuUMAP
            return cuUMAP(n_components=dim, n_neighbors=n_neighbors,
                          min_dist=min_dist, random_state=random_state)
        from umap import UMAP
        return UMAP(n_components=dim, metric="euclidean", min_dist=min_dist,
                    n_neighbors=n_neighbors, random_state=random_state)

    if mode == "vae":
        from metabo.vae import VAEEmbedder  # torch is optional
        return VAEEmbedder(latent_dim=dim, random_state=random_state)

    raise ValueError(f"unknown embedding mode: {mode}")
