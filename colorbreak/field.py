"""Smooth Gaussian random fields in 3-D via random Fourier features.

``f(x) = sqrt(2/K) * sum_k cos(w_k . x + phi_k)`` with ``w_k ~ N(0, I / ell^2)`` and
``phi_k ~ U(0, 2 pi)`` is a stationary Gaussian field with squared-exponential
covariance ``exp(-|dx|^2 / (2 ell^2))`` and exactly standard-normal marginals (in the
limit of many features; with K >= 128 the marginal is Gaussian to a few percent, and
we re-standardise empirically anyway).
"""

from __future__ import annotations

import numpy as np


class GaussianField:
    def __init__(self, length_scale: float, rng: np.random.Generator, n_features: int = 192):
        self.ell = float(length_scale)
        self.w = rng.normal(0.0, 1.0 / self.ell, size=(n_features, 3))
        self.phi = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
        self.scale = np.sqrt(2.0 / n_features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.empty(len(x))
        # chunked to keep memory bounded for large point sets
        step = 65536
        for i in range(0, len(x), step):
            proj = x[i : i + step] @ self.w.T + self.phi
            out[i : i + step] = self.scale * np.cos(proj).sum(axis=1)
        return out


class MultiOctaveField:
    """Sum of octaves (ell, ell/2, ell/4, ...) with weights (1, 1/2, 1/4, ...), standardised."""

    def __init__(self, length_scale: float, rng: np.random.Generator, octaves: int = 3, n_features: int = 192):
        self.fields = []
        self.weights = []
        for o in range(octaves):
            self.fields.append(GaussianField(length_scale / 2**o, rng, n_features))
            self.weights.append(0.5**o)
        self.norm = 1.0 / np.sqrt(sum(w * w for w in self.weights))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        acc = np.zeros(len(x))
        for f, w in zip(self.fields, self.weights):
            acc += w * f(x)
        return acc * self.norm


def standardise(v: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Shift/scale to zero (weighted) mean and unit (weighted) variance."""
    if weights is None:
        m, s = v.mean(), v.std()
    else:
        w = weights / weights.sum()
        m = float((w * v).sum())
        s = float(np.sqrt((w * (v - m) ** 2).sum()))
    return (v - m) / (s if s > 0 else 1.0)
