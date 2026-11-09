"""Mineral-specific GPR with SSA optimization of five-fold CV RMSE."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.model_selection import KFold

from .ssa import SSAOptimizer


@dataclass
class WeightedGPR:
    seed: int = 142
    use_ssa: bool = True
    population: int = 30
    iterations: int = 50
    discoverer_ratio: float = 0.25
    warning_threshold: float = 0.8
    cv_folds: int = 5
    alpha: float = 1e-10  # Numerical jitter, not the fitted noise variance.
    kernel_names: tuple[str, ...] = ("RBF", "Matern32", "Matern52", "WRBF")
    length_bounds: tuple[float, float] = (0.1, 2.0)
    signal_bounds: tuple[float, float] = (0.01, 2.0)
    noise_bounds: tuple[float, float] = (0.001, 0.1)

    @staticmethod
    def _kernel(name: str, length: float, signal: float, noise: float):
        amplitude = ConstantKernel(signal, constant_value_bounds="fixed")
        if name in ("RBF", "WRBF"):
            spatial = RBF(length, length_scale_bounds="fixed")
        elif name == "Matern32":
            spatial = Matern(length, length_scale_bounds="fixed", nu=1.5)
        elif name == "Matern52":
            spatial = Matern(length, length_scale_bounds="fixed", nu=2.5)
        else:
            raise ValueError(f"Unknown kernel: {name}")
        return amplitude * spatial + WhiteKernel(noise, noise_level_bounds="fixed")

    def kernels(self, n_features: int | None = None):
        """Kernel templates; all share one length scale, as in the manuscript."""
        return {name: self._kernel(name, 1.0, 1.0, 0.05) for name in self.kernel_names}

    def _features(self, X: np.ndarray, name: str) -> np.ndarray:
        if name == "WRBF":
            return X * np.sqrt(self.latent_weights_)
        return X

    def _fixed_gpr(self, name, params):
        return GaussianProcessRegressor(
            kernel=self._kernel(name, *params), optimizer=None,
            alpha=self.alpha, normalize_y=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, *, latent_weights=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        if X.ndim != 2 or y.ndim != 2 or len(X) != len(y):
            raise ValueError("X and y must have matching 2-D sample arrays")
        if len(X) < self.cv_folds or self.cv_folds < 2:
            raise ValueError("At least cv_folds training samples are required")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must be finite")
        valid = {"RBF", "Matern32", "Matern52", "WRBF"}
        if not self.kernel_names or set(self.kernel_names) - valid:
            raise ValueError("Unknown or empty kernel candidates")
        if latent_weights is None:
            if "WRBF" in self.kernel_names:
                raise ValueError("WRBF requires PLS latent weights from the training set")
            weights = np.ones(X.shape[1]) / X.shape[1]
        else:
            weights = np.asarray(latent_weights, float)
            if weights.shape != (X.shape[1],) or np.any(weights < 0) or not np.isfinite(weights).all():
                raise ValueError("Invalid PLS weights")
            if weights.sum() <= 0:
                raise ValueError("Sum of PLS weights must be positive")
            weights = weights / weights.sum()
        self.latent_weights_ = weights
        self.n_features_in_ = X.shape[1]

        bounds = np.array([self.length_bounds, self.signal_bounds, self.noise_bounds], dtype=float)
        if np.any(bounds[:, 0] <= 0) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("WGPR parameter bounds must be positive increasing intervals")
        initial = np.clip([1.0, 1.0, 0.05], bounds[:, 0], bounds[:, 1])
        folds = list(KFold(n_splits=self.cv_folds, shuffle=True,
                           random_state=self.seed).split(X))
        self.models_, self.kernel_choices_, self.cv_rmse_, self.hyperparameters_ = [], [], [], []
        self.candidate_results_ = []

        for target in range(y.shape[1]):
            best_kernel, best_params, best_rmse = None, None, np.inf
            for kernel_index, name in enumerate(self.kernel_names):
                inputs = self._features(X, name)

                def cv_rmse(params):
                    squared_errors = []
                    for tr, va in folds:
                        gpr = self._fixed_gpr(name, params)
                        gpr.fit(inputs[tr], y[tr, target])
                        pred = gpr.predict(inputs[va])
                        squared_errors.extend((pred - y[va, target]) ** 2)
                    return float(np.sqrt(np.mean(squared_errors)))

                if self.use_ssa:
                    optimizer = SSAOptimizer(
                        population=self.population,
                        iterations=self.iterations,
                        discoverer_ratio=self.discoverer_ratio,
                        warning_threshold=self.warning_threshold,
                        seed=self.seed + 100 * target + kernel_index,
                    )
                    params, score = optimizer.minimize(cv_rmse, bounds, initial)
                else:
                    opt = minimize(cv_rmse, initial, bounds=[tuple(b) for b in bounds],
                                   method="Nelder-Mead", options={"maxiter": 100})
                    params, score = np.clip(opt.x, bounds[:, 0], bounds[:, 1]), float(cv_rmse(np.clip(opt.x, bounds[:, 0], bounds[:, 1])))
                self.candidate_results_.append({
                    "target_index": target, "kernel": name, "cv_rmse": score,
                    "length_scale": params[0], "signal_variance": params[1],
                    "noise_variance": params[2],
                })
                if score < best_rmse:
                    best_kernel, best_params, best_rmse = name, params.copy(), float(score)

            if best_kernel is None:
                raise RuntimeError(f"No valid kernel for target {target + 1}")
            model = self._fixed_gpr(best_kernel, best_params)
            model.fit(self._features(X, best_kernel), y[:, target])
            self.models_.append(model)
            self.kernel_choices_.append(best_kernel)
            self.cv_rmse_.append(best_rmse)
            self.hyperparameters_.append(best_params)
            print(f"WGPR target {target + 1}: {best_kernel}, five-fold CV RMSE={best_rmse:.6f}", flush=True)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "models_"):
            raise RuntimeError("Fit WeightedGPR before prediction")
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Invalid PLS latent score input shape")
        means, stds = [], []
        for name, gpr in zip(self.kernel_choices_, self.models_):
            X_use = self._features(X, name)
            signal_kernel = gpr.kernel_.k1
            cross_cov = signal_kernel(X_use, gpr.X_train_)
            means.append(gpr.predict(X_use))
            v = solve_triangular(gpr.L_, cross_cov.T, lower=True, check_finite=False)
            variance = signal_kernel.diag(X_use) - np.sum(v ** 2, axis=0)
            stds.append(np.sqrt(np.maximum(variance, 0)))
        return np.column_stack(means), np.column_stack(stds)

    def parameter_table(self, names: list[str]) -> pd.DataFrame:
        if len(names) != len(self.models_):
            raise ValueError("Mineral name count does not match trained models")
        return pd.DataFrame([
            {
                "mineral": mineral,
                "kernel": kind,
                "length_scale": float(theta[0]),
                "signal_variance": float(theta[1]),
                "noise_variance": float(theta[2]),
                "cv_rmse": float(rmse),
                "cv_mse": float(rmse ** 2),
                "n_folds": self.cv_folds,
            }
            for mineral, kind, theta, rmse in zip(
                names, self.kernel_choices_, self.hyperparameters_, self.cv_rmse_)
        ])
