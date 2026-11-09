"""PLS latent variable selection and variance-based latent weights."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from ._cv import cv_mse, PLS_MAX_ITER, PLS_TOL


@dataclass
class PLSFeatures:
    max_components: int = 50
    evaluator: str = "pls"
    seed: int = 142

    def fit(self, X: np.ndarray, y: np.ndarray, folds, *, X_for_cv=None):
        # X is already standardized using train-only spectral scalers.
        # X_for_cv is the selected log spectrum, not fitted before CV.
        X_cv = X if X_for_cv is None else X_for_cv
        max_allowed = min(self.max_components, X.shape[1],
                          min(len(train) for train, _ in folds) - 1)
        if max_allowed < 1:
            raise ValueError("Insufficient samples/features for PLS")
        scores = []
        for n in range(1, max_allowed + 1):
            mse = cv_mse(X_cv, y, folds, n_components=n, evaluator=self.evaluator,
                         stage="pls", seed=self.seed)
            scores.append({"n_components": n, "cv_mse": mse, "rmsecv": float(np.sqrt(mse))})
        self.cv_results_ = pd.DataFrame(scores)
        self.n_components_ = int(self.cv_results_.loc[self.cv_results_["cv_mse"].idxmin(), "n_components"])
        self.pls_ = PLSRegression(
            n_components=self.n_components_, max_iter=PLS_MAX_ITER, tol=PLS_TOL
        ).fit(X, y)
        latent_scores = self.pls_.transform(X)
        var = latent_scores.var(axis=0)
        if var.sum() <= 0:
            raise ValueError("PLS latent variables have zero variance")
        self.weights_ = var / var.sum()
        # Apply variance-based weights to the PLS scores as defined in the paper.
        # Standardizing each score first would cancel the variance weighting.
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project inputs into the unweighted PLS latent space."""
        if not hasattr(self, "pls_"):
            raise RuntimeError("Fit PLSFeatures before transforming")
        return self.pls_.transform(X)

    def transform_weighted(self, X: np.ndarray) -> np.ndarray:
        """Expose variance-weighted scores without modifying the model input."""
        return self.transform(X) * np.sqrt(self.weights_)
