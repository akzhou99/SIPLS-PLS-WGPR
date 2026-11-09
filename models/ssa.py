"""Sparrow-search population optimizer for a bounded scalar objective."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SSAOptimizer:
    population: int = 30
    iterations: int = 50
    discoverer_ratio: float = 0.25
    warning_threshold: float = 0.8
    scout_ratio: float = 0.15
    seed: int = 142

    def minimize(self, objective: Callable[[np.ndarray], float], bounds,
                 initial=None) -> tuple[np.ndarray, float]:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_parameters, 2)")
        lower, upper = bounds[:, 0], bounds[:, 1]
        if not np.all(np.isfinite(bounds)) or np.any(lower >= upper):
            raise ValueError("SSA requires finite lower bounds < upper bounds")
        if self.population < 4 or self.iterations < 1:
            raise ValueError("SSA requires population >= 4 and iterations >= 1")
        if not (0 < self.discoverer_ratio < 1 and 0 < self.scout_ratio < 1
                and 0 < self.warning_threshold < 1):
            raise ValueError("SSA ratios and warning threshold must be in (0, 1)")

        rng = np.random.default_rng(self.seed)
        n, d = self.population, len(lower)
        span = upper - lower
        positions = rng.uniform(lower, upper, (n, d))
        if initial is not None:
            initial = np.asarray(initial, dtype=float)
            if initial.shape != (d,) or not np.isfinite(initial).all():
                raise ValueError("initial must contain one finite value per parameter")
            positions[0] = np.clip(initial, lower, upper)

        def evaluate(point):
            try:
                val = float(objective(point))
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                return np.inf
            return val if np.isfinite(val) else np.inf

        fitness = np.array([evaluate(x) for x in positions])
        best_idx = int(np.argmin(fitness))
        best_position, best_value = positions[best_idx].copy(), float(fitness[best_idx])
        n_discoverers = max(1, int(n * self.discoverer_ratio))
        n_scouts = max(1, int(n * self.scout_ratio))

        for iteration in range(self.iterations):
            order = np.argsort(fitness)
            positions, fitness = positions[order], fitness[order]
            current_best = positions[0].copy()
            worst = positions[-1].copy()
            updated = positions.copy()
            shrink = 1.0 - iteration / self.iterations
            alarm = rng.random()

            for i in range(n_discoverers):
                scale = 0.15 if alarm < self.warning_threshold else 0.35
                updated[i] += rng.normal(size=d) * scale * span * shrink
            for i in range(n_discoverers, n):
                if i >= n / 2:
                    updated[i] += rng.normal(size=d) * 0.2 * np.abs(positions[i] - worst)
                else:
                    updated[i] = current_best + rng.normal(size=d) * 0.2 * np.abs(positions[i] - current_best)
            for i in rng.choice(n, size=n_scouts, replace=False):
                if fitness[i] > fitness[0]:
                    updated[i] = current_best + rng.normal(size=d) * 0.1 * span
                else:
                    updated[i] += rng.normal(size=d) * 0.1 * (np.abs(positions[i] - worst) + 1e-8)

            updated = np.clip(updated, lower, upper)
            updated_fitness = np.array([evaluate(x) for x in updated])
            improved = updated_fitness < fitness
            positions[improved] = updated[improved]
            fitness[improved] = updated_fitness[improved]
            index = int(np.argmin(fitness))
            if fitness[index] < best_value:
                best_value = float(fitness[index])
                best_position = positions[index].copy()

        if not np.isfinite(best_value):
            raise RuntimeError("SSA did not find a finite objective value")
        return best_position, best_value

    def __call__(self, objective, initial_theta, bounds):
        return self.minimize(lambda theta: objective(theta, eval_gradient=False),
                             bounds, initial_theta)
