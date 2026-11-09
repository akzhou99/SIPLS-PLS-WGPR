"""Train-fold-only evaluation used for SIPLS and PLS model selection."""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# NIPALS convergence settings shared by CV and the final PLS model.
PLS_MAX_ITER = 2000
PLS_TOL = 1e-6


def make_folds(n_samples: int, n_splits: int, seed: int):
    if n_splits < 2 or n_samples < n_splits:
        raise ValueError(f"CV requires at least {n_splits} samples for {n_splits} folds; got {n_samples}")
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n_samples)))


def cv_mse(X: np.ndarray, y: np.ndarray, folds, *, n_components: int,
           evaluator: str, stage: str, seed: int) -> float:
    predictions = np.zeros_like(y, dtype=float)
    for train, valid in folds:
        scaler = StandardScaler().fit(X[train])
        X_train, X_val = scaler.transform(X[train]), scaler.transform(X[valid])
        lv = min(n_components, X_train.shape[1], len(train) - 1)
        if lv < 1:
            raise ValueError("Insufficient samples for PLS fitting")
        if evaluator == "xgboost" and stage == "sipls" and X_train.shape[1] <= n_components + 1:
            z_train, z_val = X_train, X_val
        else:
            pls = PLSRegression(n_components=lv, max_iter=PLS_MAX_ITER, tol=PLS_TOL)
            pls.fit(X_train, y[train])
            if evaluator == "pls":
                predictions[valid] = pls.predict(X_val)
                continue
            z_train, z_val = pls.transform(X_train), pls.transform(X_val)
        if evaluator != "xgboost":
            raise ValueError("evaluator must be 'xgboost' or 'pls'")
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        options = (dict(n_estimators=100, learning_rate=0.05, max_depth=3,
                        subsample=0.8, colsample_bytree=0.8)
                   if stage == "sipls" else
                   dict(n_estimators=300, learning_rate=0.1, max_depth=4,
                        subsample=0.9, colsample_bytree=0.9))
        model = MultiOutputRegressor(XGBRegressor(
            objective="reg:squarederror", tree_method="hist", n_jobs=1,
            random_state=seed, verbosity=0, **options))
        model.fit(z_train, y[train])
        predictions[valid] = model.predict(z_val)
    return float(mean_squared_error(y, predictions))
