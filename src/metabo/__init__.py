"""Benchmark of PCA, UMAP and VAE embeddings for prediction from the plasma metabolome."""
import warnings

# The elastic-net LogisticRegression is built for scikit-learn 1.4 to 1.9; 1.8 deprecates
# the `penalty` argument that older versions require.
warnings.filterwarnings("ignore", message=".*penalty.*", category=FutureWarning)
