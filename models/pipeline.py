"""End-to-end SIPLS–PLS–SSA-WGPR estimator."""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from ._cv import make_folds
from .pls import PLSFeatures
from .sipls import SIPLSSelector
from .wgpr import WeightedGPR
from .data import SpectralPreprocessor, normalize_targets, simplex_projection


@dataclass
class SIPLSPLSSSAWGPR:
    schemes: tuple[tuple[int, int], ...] = ((8, 6), (16, 12), (32, 24), (64, 32))
    sipls_components: int = 10
    pls_max_components: int = 50
    evaluator: str = "pls"
    cv: int = 10
    pls_cv: int = 5
    spectrum_scale: float = 1e6
    seed: int = 142
    wgpr: WeightedGPR = field(default_factory=WeightedGPR)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all stages on training wells only; X columns must be inelastic then capture."""
        X = self._check_features(X)
        y, self.target_scale_, self.closed_composition_ = normalize_targets(y)
        if len(X) != len(y):
            raise ValueError("X and y have different sample counts")
        folds = make_folds(len(X), self.cv, self.seed)
        pls_folds = make_folds(len(X), self.pls_cv, self.seed)
        self.inelastic_ = SpectralPreprocessor(self.spectrum_scale)
        self.capture_ = SpectralPreprocessor(self.spectrum_scale)
        block_i, block_c = X[:, :514], X[:, 514:]
        log_i, log_c = self.inelastic_.log_transform(block_i), self.capture_.log_transform(block_c)
        self.sipls_i_ = SIPLSSelector(self.schemes, self.sipls_components, self.evaluator, self.seed).fit(log_i, y, folds)
        self.sipls_c_ = SIPLSSelector(self.schemes, self.sipls_components, self.evaluator, self.seed).fit(log_c, y, folds)
        Xi = self.inelastic_.fit_transform(block_i)
        Xc = self.capture_.fit_transform(block_c)
        selected = np.column_stack((self.sipls_i_.transform(Xi), self.sipls_c_.transform(Xc)))
        selected_logs = np.column_stack((self.sipls_i_.transform(log_i), self.sipls_c_.transform(log_c)))
        self.pls_ = PLSFeatures(self.pls_max_components, self.evaluator, self.seed).fit(
            selected, y, pls_folds, X_for_cv=selected_logs)
        self.wgpr.fit(self.pls_.transform(selected), y, latent_weights=self.pls_.weights_)
        self.n_features_in_ = X.shape[1]
        self.n_targets_ = y.shape[1]
        return self

    @staticmethod
    def _check_features(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 1028 or not np.isfinite(X).all():
            raise ValueError("X must be a finite (n_samples, 1028) matrix")
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pls_"):
            raise RuntimeError("Fit the model before calling transform")
        X = self._check_features(X)
        Xi = self.sipls_i_.transform(self.inelastic_.transform(X[:, :514]))
        Xc = self.sipls_c_.transform(self.capture_.transform(X[:, 514:]))
        return self.pls_.transform(np.column_stack((Xi, Xc)))

    def transform_weighted(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X) * np.sqrt(self.pls_.weights_)

    def predict(self, X: np.ndarray, *, constrain: bool = True) -> tuple[np.ndarray, np.ndarray]:
        raw_scores = self.transform(X)
        mean, std = self.wgpr.predict(raw_scores)
        if constrain:
            mean = simplex_projection(mean) if self.closed_composition_ else np.clip(mean, 0, None)
        return mean * self.target_scale_, std * self.target_scale_

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path):
        model = joblib.load(path)  # Only load model files from trusted sources.
        if not isinstance(model, cls):
            raise TypeError("Model file is not a SIPLSPLSSSAWGPR instance")
        return model
