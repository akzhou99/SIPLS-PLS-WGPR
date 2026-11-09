"""Forward selection of synergistic spectral intervals."""

from dataclasses import dataclass

import numpy as np

from ._cv import cv_mse


@dataclass
class SIPLSSelector:
    schemes: tuple[tuple[int, int], ...] = ((8, 6), (16, 12), (32, 24), (64, 32))
    n_components: int = 10
    evaluator: str = "pls"
    seed: int = 142
    tolerance: float = 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray, folds):
        if X.ndim != 2 or X.shape[1] != 512:
            raise ValueError("SIPLS expects a spectrum with 512 effective channels")
        self.results_ = []
        for count, limit in self.schemes:
            if 512 % count:
                raise ValueError("Number of intervals must divide 512")
            width = 512 // count
            remaining = [(i * width, (i + 1) * width) for i in range(count)]
            chosen = []
            best_mse = np.inf
            for _ in range(min(limit, count)):
                candidate, candidate_mse = None, np.inf
                for interval in remaining:
                    indices = [j for a, b in chosen + [interval] for j in range(a, b)]
                    mse = cv_mse(X[:, indices], y, folds, n_components=self.n_components,
                                 evaluator=self.evaluator, stage="sipls", seed=self.seed)
                    if mse < candidate_mse:
                        candidate, candidate_mse = interval, mse
                if candidate is None or candidate_mse >= best_mse - self.tolerance:
                    break
                chosen.append(candidate)
                remaining.remove(candidate)
                best_mse = candidate_mse
            if not chosen:
                raise RuntimeError("No spectral intervals were selected")
            result = {
                "n_intervals": count, "width": width, "max_selected": limit,
                "selected": len(chosen), "selected_intervals": chosen,
                "n_channels": len(chosen) * width,
                "cv_mse": best_mse, "rmsecv": float(np.sqrt(best_mse)),
            }
            self.results_.append(result)
            print(f"SIPLS {count} intervals: selected {len(chosen)}, RMSECV={np.sqrt(best_mse):.6f}", flush=True)
        self.best_ = min(self.results_, key=lambda item: item["cv_mse"])
        self.indices_ = np.asarray([j for a, b in self.best_["selected_intervals"]
                                    for j in range(a, b)], dtype=int)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "indices_"):
            raise RuntimeError("Fit SIPLSSelector before transforming")
        return np.asarray(X)[:, self.indices_]
