"""Fixed-format Excel input, well splitting and spectral preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

N_CHANNELS = 514
N_FEATURES = 2 * N_CHANNELS
N_TARGETS = 8


@dataclass
class Dataset:
    depth: np.ndarray
    X: np.ndarray
    y: np.ndarray | None
    mineral_names: list[str] | None
    source_sheet: np.ndarray
    well_id: np.ndarray

def check_splits(train: list[str], test: list[str], external: list[str]) -> None:
    if not train:
        raise ValueError("At least one training worksheet must be specified")
    groups = [set(train), set(test), set(external)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Training, test and external worksheets must not overlap")

def check_well_splits(train: list[str], test: list[str], external: list[str]) -> None:
    if (len(train), len(test), len(external)) != (6, 2, 2):
        raise ValueError("Specify 6 training, 2 test and 2 external-validation wells")
    names = [str(name).strip() for group in (train, test, external) for name in group]
    if any(not name for name in names) or len(set(names)) != 10:
        raise ValueError("The ten well names must be nonempty and unique")

def subset_wells(data: Dataset, names: list[str]) -> Dataset:
    requested = {str(name).strip() for name in names}
    available = set(data.well_id)
    missing = requested - available
    if missing:
        raise ValueError(f"Wells not found in workbook: {sorted(missing)}")
    mask = np.isin(data.well_id, list(requested))
    return Dataset(
        depth=data.depth[mask],
        X=data.X[mask],
        y=data.y[mask] if data.y is not None else None,
        mineral_names=data.mineral_names,
        source_sheet=data.source_sheet[mask],
        well_id=data.well_id[mask],
    )

def read_workbook(
    path: str | Path,
    sheets: list[str],
    *,
    order: str = "inelastic-first",
    require_labels: bool = True,
) -> Dataset:
    """Read the fixed workbook layout used by this project.

    Training: well name | depth | 1028 spectral channels | 8 mineral columns (1038 columns).
    Prediction: well name | depth | 1028 spectral channels (1030 columns).
    """
    if not sheets:
        raise ValueError("No worksheet names supplied")
    if order not in ("inelastic-first", "capture-first"):
        raise ValueError("SPECTRA_ORDER must be 'inelastic-first' or 'capture-first'")

    book = pd.ExcelFile(path)
    missing_sheets = [sheet for sheet in sheets if sheet not in book.sheet_names]
    if missing_sheets:
        raise ValueError(f"Worksheets not found: {missing_sheets}")

    depths, spectra, labels, wells, sources = [], [], [], [], []
    mineral_names = None
    label_presence = []

    for sheet in sheets:
        frame = pd.read_excel(book, sheet_name=sheet)
        n_columns = frame.shape[1]
        has_labels = n_columns == 1038
        if n_columns not in (1030, 1038):
            raise ValueError(
                f"{sheet}: found {n_columns} columns; expected 1038 for training "
                "(well name + depth + 1028 channels + 8 minerals), or 1030 for prediction "
                "(well name + depth + 1028 channels). See INPUT_FORMAT.md."
            )
        if require_labels and not has_labels:
            raise ValueError(f"{sheet}: training data must include the final eight mineral columns")

        well = frame.iloc[:, 0].astype("string").str.strip()
        if well.isna().any() or (well == "").any():
            raise ValueError(f"{sheet}: missing well name in the first column")
        depth = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        X = frame.iloc[:, 2:2 + N_FEATURES].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if X.shape[1] != N_FEATURES:
            raise ValueError(f"{sheet}: expected exactly 1028 spectral channels")
        if not np.isfinite(depth).all() or not np.isfinite(X).all():
            raise ValueError(f"{sheet}: depth or spectral channels contain missing/non-numeric values")

        if order == "capture-first":
            X = np.column_stack((X[:, N_CHANNELS:], X[:, :N_CHANNELS]))

        if has_labels:
            names = [str(column).strip() for column in frame.columns[-N_TARGETS:]]
            if mineral_names is None:
                mineral_names = names
            elif names != mineral_names:
                raise ValueError(f"{sheet}: the final eight mineral column names differ between sheets")
            y = frame.iloc[:, -N_TARGETS:].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            if require_labels:
                if not np.isfinite(y).all():
                    raise ValueError(f"{sheet}: mineral labels contain missing/non-numeric values")
                if (y < 0).any():
                    raise ValueError(f"{sheet}: mineral contents must be nonnegative")
                if (y.sum(axis=1) == 0).any():
                    raise ValueError(f"{sheet}: found rows where all eight mineral contents are zero")
            elif not np.isfinite(y).all():
                # For prediction workbooks, blank target columns are treated as no reference labels.
                if np.isfinite(y).any():
                    raise ValueError(f"{sheet}: mineral labels are only partially filled")
                y = None
        else:
            y = None

        depths.append(depth)
        spectra.append(X)
        wells.extend(well.astype(str).to_list())
        sources.extend([sheet] * len(frame))
        labels.append(y)
        label_presence.append(y is not None)

    if any(label_presence) and not all(label_presence):
        raise ValueError("Do not mix workbooks with and without mineral labels in one read")

    y_all = np.vstack(labels) if all(label_presence) else None
    return Dataset(
        depth=np.concatenate(depths),
        X=np.vstack(spectra),
        y=y_all,
        mineral_names=mineral_names,
        source_sheet=np.asarray(sources, dtype=str),
        well_id=np.asarray(wells, dtype=str),
    )

class SpectralPreprocessor:
    def __init__(self, scale: float = 1e6):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Spectrum normalization scale must be positive and finite")
        self.scale = scale
        self.scaler = StandardScaler()

    def log_transform(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=float)
        if block.ndim != 2 or block.shape[1] != 514:
            raise ValueError("Each net spectrum must contain 514 channels")
        if not np.isfinite(block).all():
            raise ValueError("Non-finite spectral count")
        X = np.maximum(block[:, 1:-1], 0.0)
        totals = X.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise ValueError("A spectrum has zero total counts after edge removal")
        return np.log1p(X / totals * self.scale)

    def fit(self, block: np.ndarray):
        self.scaler.fit(self.log_transform(block))
        return self

    def transform(self, block: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.log_transform(block))

    def fit_transform(self, block: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(self.log_transform(block))

def normalize_targets(y: np.ndarray) -> tuple[np.ndarray, float, bool]:
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] != 8 or not np.isfinite(y).all():
        raise ValueError("Training labels must be a finite n-by-8 array")
    if (y < 0).any():
        raise ValueError("Mineral contents must be nonnegative")
    row_sums = y.sum(axis=1)
    has_percent = (np.abs(row_sums - 100) <= 5).any()
    has_fraction = (np.abs(row_sums - 1) <= 0.05).any()
    if has_percent and has_fraction:
        raise ValueError("Training labels mix percentage (0-100) and fraction (0-1) units")
    if np.median(np.abs(row_sums - 100)) <= 5:
        return y / 100, 100.0, True
    if np.median(np.abs(row_sums - 1)) <= 0.05:
        return y.copy(), 1.0, True
    return y.copy(), 1.0, False

def simplex_projection(y: np.ndarray) -> np.ndarray:
    """Euclidean projection of each prediction row onto the probability simplex."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    for i, v in enumerate(y):
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        rho = np.flatnonzero(u - cssv / np.arange(1, len(u) + 1) > 0)
        theta = cssv[rho[-1]] / (rho[-1] + 1) if rho.size else 0.0
        projected = np.maximum(v - theta, 0)
        out[i] = projected / projected.sum() if projected.sum() else np.full(len(v), 1 / len(v))
    return out
