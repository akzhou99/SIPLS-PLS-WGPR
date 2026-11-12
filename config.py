"""Paths, well partitions, and model parameters."""

from pathlib import Path
import os

DATA_FILE = Path(r"E:\example\Input data.xlsx")

TRAIN_SHEETS = ["Train well"]
TEST_SHEETS = ["Test well"]
EXTERNAL_SHEETS = []

TRAIN_WELLS = []
TEST_WELLS = []
EXTERNAL_WELLS = []

# Fixed input layout is documented in INPUT_FORMAT.md.
SPECTRA_ORDER = "inelastic-first"  # or "capture-first" for unnamed legacy data
RANDOM_STATE = 142
CV_FOLDS = 10
PLS_CV_FOLDS = 5
SIPLS_SCHEMES = ((8, 6), (16, 12), (32, 24), (64, 32))
SIPLS_COMPONENTS = 10
PLS_MAX_COMPONENTS = 50
EVALUATOR = "pls"  # PLS regression evaluates spectral intervals and latent variables
SPECTRUM_SCALE = 1e6

# SSA and WGPR settings
SSA_POPULATION = 30
SSA_ITERATIONS = 50
SSA_DISCOVERER_RATIO = 0.25
SSA_WARNING_THRESHOLD = 0.8
WGPR_CV_FOLDS = 5
WGPR_ALPHA = 1e-10  # numerical jitter; sigma_n^2 is optimized separately
OUTPUT_DIR = Path("results_paper_method")
