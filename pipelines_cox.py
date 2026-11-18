"""Preprocessing transformers and the penalised Cox pipeline."""
from __future__ import annotations

import warnings

import numpy as np
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis

import config_cox as cfg

_MEMORY = Memory(location=str(cfg.CACHE_DIR), verbose=0) if cfg.USE_CACHE else None


class VarianceFilter(BaseEstimator, TransformerMixin):
    """Drop features that are constant, or numerically constant, in the training data."""

    def __init__(self, percentile: float = cfg.VARIANCE_PERCENTILE,
                 tolerance: float = 1e-12):
        self.percentile = percentile
        self.tolerance = tolerance

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        scale = np.maximum(np.abs(np.mean(X, axis=0)), 1.0) ** 2
        self.mask_ = np.var(X, axis=0) > self.tolerance * scale
        if self.mask_.sum() == 0:
            self.mask_ = np.ones(X.shape[1], dtype=bool)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float)[:, self.mask_]

    def get_support(self, indices: bool = False):
        return np.where(self.mask_)[0] if indices else self.mask_


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove one member of each near-duplicate pair."""

    def __init__(self, threshold: float = cfg.CORRELATION_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)

        variances = np.var(X, axis=0)
        dropped: set[int] = set()
        rows, cols = np.where(np.triu(corr > self.threshold, k=1))
        for i, j in zip(rows, cols):
            if i in dropped or j in dropped:
                continue
            dropped.add(int(j) if variances[i] >= variances[j] else int(i))

        self.to_keep_ = [i for i in range(n) if i not in dropped]
        self.n_features_in_ = n
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float)[:, self.to_keep_]

    def get_support(self, indices: bool = False):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.to_keep_] = True
        return np.where(mask)[0] if indices else mask


class PartialScaler(BaseEstimator, TransformerMixin):
    """Standardise continuous columns, leave binary ones untouched."""

    def __init__(self, binary_indices: list[int] | None = None):
        self.binary_indices = binary_indices

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        binary = self.binary_indices or []
        for i in binary:
            if i >= X.shape[1]:
                raise ValueError(f"binary index {i} out of range: an earlier "
                                 f"step removed columns")
            values = np.unique(X[:, i][~np.isnan(X[:, i])])
            if not set(values.tolist()) <= {0.0, 1.0}:
                raise ValueError(f"column {i} flagged binary but takes values "
                                 f"{values[:5]}: column positions have shifted")
        self.continuous_ = [i for i in range(X.shape[1]) if i not in binary]
        self.scaler_ = StandardScaler()
        if self.continuous_:
            self.scaler_.fit(X[:, self.continuous_])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        out = np.asarray(X, dtype=float).copy()
        if self.continuous_:
            out[:, self.continuous_] = self.scaler_.transform(
                out[:, self.continuous_])
        return out


def build_pipeline(binary_indices=None, l1_ratio=0.9, alpha=None,
                   alphas=None) -> Pipeline:
    """Preprocessing followed by a penalised Cox model."""
    model = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        alpha_min_ratio=cfg.ALPHA_MIN_RATIO,
        n_alphas=cfg.N_ALPHAS,
        alphas=(list(alphas) if alphas is not None
                else [alpha] if alpha is not None else None),
        fit_baseline_model=True,
        max_iter=100_000,
    )

    return Pipeline(
        [
            ("variance", VarianceFilter()),
            ("scaler", PartialScaler(binary_indices=binary_indices)),
            ("correlation", CorrelationFilter()),
            ("coxnet", model),
        ],
        memory=_MEMORY,
    )


def selected_features(pipeline: Pipeline, feature_names: list[str],
                      alpha: float | None = None) -> dict[str, float]:
    """Non-zero coefficients of a fitted pipeline, mapped back to the original feature names."""
    model = pipeline.named_steps["coxnet"]

    indices = np.arange(len(feature_names))
    indices = indices[pipeline.named_steps["variance"].get_support()]
    indices = indices[pipeline.named_steps["correlation"].get_support()]

    coefficients = model.coef_
    if coefficients.ndim > 1:
        if alpha is not None and hasattr(model, "alphas_"):
            column = int(np.argmin(np.abs(model.alphas_ - alpha)))
        else:
            column = coefficients.shape[1] - 1
        coefficients = coefficients[:, column]

    non_zero = np.flatnonzero(coefficients)
    return {feature_names[indices[i]]: float(coefficients[i]) for i in non_zero}


def risk_scores(pipeline: Pipeline, X, alpha: float | None = None) -> np.ndarray:
    """Linear predictor for each patient: the radiomic score."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if alpha is not None:
            return pipeline.predict(X, alpha=alpha)
        return pipeline.predict(X)


def fit_at_alpha(X, y, binary_indices, l1_ratio, alpha) -> Pipeline:
    """Fit a pipeline at one fixed penalty, reached along the regularisation path."""
    alpha = float(alpha)
    path = alpha_grid(X, y, binary_indices, l1_ratio)
    sequence = sorted({float(a) for a in path if a > alpha} | {alpha},
                      reverse=True)
    pipeline = build_pipeline(binary_indices, l1_ratio=l1_ratio,
                              alphas=sequence)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X, y)
    return pipeline


def alpha_grid(X, y, binary_indices, l1_ratio) -> np.ndarray:
    """Penalty values to search, taken from the path Coxnet computes itself."""
    pipeline = build_pipeline(binary_indices, l1_ratio=l1_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X, y)
    alphas = pipeline.named_steps["coxnet"].alphas_
    return np.asarray(alphas, dtype=float)
