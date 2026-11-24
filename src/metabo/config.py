"""Experiment settings.

Paths can be changed with environment variables:
  METABO_DATA_DIR     input matrices and targets (default: <repo>/data)
  METABO_RESULTS_DIR  output folder (default: <repo>/results)
  METABO_SOURCE       raw metabolome file read by build_inputs.py and the fill checks
                      (default: <data dir>/datacorr_6m6y18mMw1w24_final.csv)
  METABO_TARGET_FILE, METABO_METAB_6M, METABO_METAB_18M, METABO_METAB_6Y
                      the per-timepoint input matrices and the target file, for a data
                      folder that does not use the default filenames (e.g. the renamed
                      synthetic data in data/)
  METABO_FISHOIL_MARKER, METABO_FISHOIL_MARKER_PARENT
                      the known fish-oil markers' exact metabolite names (the strong
                      hydroxylated marker, and separately its weaker parent compound),
                      searched for by the attribution scripts (default: hydroxy-CMPF and
                      the parent CMPF compound's name prefix); set these if the data
                      folder renames metabolites to something else

For a quick test run, METABO_N_PERMUTATION, METABO_N_BOOTSTRAP, METABO_EMBED_DIMS
(e.g. "2 16") and METABO_VAE_MAX_EPOCHS override the defaults below.
"""
import os
from pathlib import Path


def _env_int(name, default):
    return int(os.environ.get(name, default))


ROOT = Path(__file__).resolve().parents[2]  # config.py -> metabo/ -> src/ -> repo root
DATA_DIR = Path(os.environ.get("METABO_DATA_DIR", ROOT / "data")).resolve()
RESULTS_DIR = Path(os.environ.get("METABO_RESULTS_DIR", ROOT / "results")).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_FILE = Path(os.environ.get("METABO_SOURCE", DATA_DIR / "datacorr_6m6y18mMw1w24_final.csv"))

METAB_FILES = {
    "6m": Path(os.environ.get("METABO_METAB_6M", DATA_DIR / "metabolites_6m_for_ml.csv")),
    "18m": Path(os.environ.get("METABO_METAB_18M", DATA_DIR / "metabolites_18m_for_ml.csv")),
    "6y": Path(os.environ.get("METABO_METAB_6Y", DATA_DIR / "metabolites_6y_for_ml.csv")),
}
TARGET_FILE = Path(os.environ.get("METABO_TARGET_FILE", DATA_DIR / "target_for_metabolites_ml.csv"))
FISHOIL_MARKER = os.environ.get("METABO_FISHOIL_MARKER", "hydroxy-CMPF")
FISHOIL_MARKER_PARENT = os.environ.get("METABO_FISHOIL_MARKER_PARENT", "3-carboxy-4-methyl-5-propyl")

TIMEPOINTS = ["6m", "18m", "6y"]
REPRESENTATIONS = ["raw", "pca", "umap", "vae"]
EMBED_DIMS = [int(k) for k in os.environ.get("METABO_EMBED_DIMS", "2 3 4 8 16").split()]
MODELS = ["lr", "rfc", "knn"]
SEED = 42

# Outcomes analyzed at each timepoint; sex is the positive control.
ROSTER = [
    "AD 6 yr ever", "Asthma 6 yr ever", "atopy_temp", "sens_s_ig_e_any",
    "sens_inhal", "sens_food", "ar_6y",
    "fetal-Antibiotics during pregnancy",
    "fetal-Antibiotics during birth", "fetal-Delivery",
    "mother_smoking", "fishoil", "vitd",
]
SEX_CONTROL = "sex"
OUTCOMES = ROSTER + [SEX_CONTROL]

OUTER_K = 5
INNER_K = 4
N_BOOTSTRAP = _env_int("METABO_N_BOOTSTRAP", 1000)
N_PERMUTATION = _env_int("METABO_N_PERMUTATION", 1000)
MIN_EVENTS = 10

# The VAE is tuned only on validation reconstruction loss, never on the outcome.
VAE_MAX_EPOCHS = _env_int("METABO_VAE_MAX_EPOCHS", 500)
VAE_PATIENCE = 20
VAE_VAL_FRACTION = 0.2
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
