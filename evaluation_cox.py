"""Repeated nested cross-validation, optimism correction, final model and model comparisons."""
from __future__ import annotations

import gc
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import config_cox as cfg
from pipelines_cox import (
    alpha_grid,
    build_pipeline,
    fit_at_alpha,
    risk_scores,
    selected_features,
)


def _append(path: Path, row: dict) -> None:
    """Append one record and force it to disk immediately."""
    header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        pd.DataFrame([row]).to_csv(fh, header=header, index=False)
        fh.flush()
        os.fsync(fh.fileno())


def _done_keys(path: Path, columns: list[str]) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set() if df.empty else set(map(tuple, df[columns].values.tolist()))


def c_index(event, duration, risk) -> float:
    """Harrell's concordance index."""
    return float(concordance_index_censored(
        event.astype(bool), duration, risk)[0])


def _tune(X, event, duration, binary_indices, seed):
    """Select the penalty by inner cross-validation."""
    inner = KFold(n_splits=cfg.N_SPLITS_INNER, shuffle=True, random_state=seed)
    best = (-np.inf, cfg.L1_RATIOS[0], None)

    for l1_ratio in cfg.L1_RATIOS:
        try:
            y_all = Surv.from_arrays(event=event.astype(bool), time=duration)
            alphas = alpha_grid(X, y_all, binary_indices, l1_ratio)
        except Exception:
            continue

        alphas = alphas[:: max(1, len(alphas) // 12)]

        scores = {a: [] for a in alphas}
        for train, test in inner.split(X):
            y_train = Surv.from_arrays(
                event=event[train].astype(bool), time=duration[train])
            try:
                pipe = build_pipeline(binary_indices, l1_ratio=l1_ratio)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X[train], y_train)
            except Exception:
                continue

            for a in alphas:
                try:
                    risk = risk_scores(pipe, X[test], alpha=a)
                    scores[a].append(
                        c_index(event[test], duration[test], risk))
                except Exception:
                    continue

        for a, values in scores.items():
            if len(values) >= 3:
                mean = float(np.mean(values))
                if mean > best[0]:
                    best = (mean, l1_ratio, float(a))

    return best[1], best[2]


def _one_fold(ds, space, train, test, seed):
    """One outer fold: tune on training rows, evaluate on held-out rows."""
    X = ds.X(space)
    binary = ds.binary_indices(space)
    names = ds.columns_for(space)

    l1_ratio, alpha = _tune(
        X[train], ds.event[train], ds.duration[train], binary, seed)
    if alpha is None:
        return None

    y_train = Surv.from_arrays(
        event=ds.event[train].astype(bool), time=ds.duration[train])
    pipeline = fit_at_alpha(X[train], y_train, binary, l1_ratio, alpha)

    risk_test = risk_scores(pipeline, X[test], alpha=alpha)
    risk_train = risk_scores(pipeline, X[train], alpha=alpha)

    coefficients = selected_features(pipeline, names, alpha=alpha)

    result = {
        "c_test": c_index(ds.event[test], ds.duration[test], risk_test),
        "c_train": c_index(ds.event[train], ds.duration[train], risk_train),
        "l1_ratio": l1_ratio,
        "alpha": alpha,
        "n_selected": len(coefficients),
        "selected": coefficients,
        "test_idx": test,
        "risk_test": risk_test,
    }
    del pipeline
    gc.collect()
    return result


def repeated_cv(ds, spaces=None, n_repeats=None, tag="main") -> pd.DataFrame:
    """Repeated nested cross-validation over the requested feature spaces."""
    spaces = spaces or list(cfg.FEATURE_SPACES)
    n_repeats = cfg.N_REPEATS if n_repeats is None else n_repeats

    fold_path = cfg.CKPT_DIR / f"cv_folds_{tag}.csv"
    risk_path = cfg.CKPT_DIR / f"cv_risk_{tag}.csv"
    sel_path = cfg.CKPT_DIR / f"cv_selected_{tag}.csv"

    done = _done_keys(fold_path, ["space", "repeat", "fold"])
    total = len(spaces) * n_repeats * cfg.N_SPLITS_OUTER
    if done:
        print(f"  resuming: {len(done)}/{total} folds already on disk")

    started, completed = time.time(), len(done)

    for space in spaces:
        for repeat in range(n_repeats):
            seed = cfg.SEED + repeat
            splitter = KFold(n_splits=cfg.N_SPLITS_OUTER, shuffle=True,
                             random_state=seed)
            splits = list(splitter.split(ds.X(space)))

            pending = [(k, tr, te) for k, (tr, te) in enumerate(splits)
                       if (space, repeat, k) not in done]
            if not pending:
                continue

            results = Parallel(n_jobs=cfg.N_JOBS)(
                delayed(_one_fold)(ds, space, tr, te, seed)
                for _, tr, te in pending)

            for (fold, _, _), res in zip(pending, results):
                if res is None:
                    _append(fold_path, {"space": space, "repeat": repeat,
                                        "fold": fold, "c_test": np.nan,
                                        "c_train": np.nan, "gap": np.nan,
                                        "l1_ratio": np.nan, "alpha": np.nan,
                                        "n_selected": 0, "status": "failed"})
                    continue

                _append(fold_path, {
                    "space": space, "repeat": repeat, "fold": fold,
                    "c_test": res["c_test"], "c_train": res["c_train"],
                    "gap": res["c_train"] - res["c_test"],
                    "l1_ratio": res["l1_ratio"], "alpha": res["alpha"],
                    "n_selected": res["n_selected"], "status": "ok",
                })
                for idx, risk in zip(res["test_idx"], res["risk_test"]):
                    _append(risk_path, {
                        "space": space, "repeat": repeat,
                        "patient_index": int(idx), "risk": float(risk)})
                for feature, coefficient in res["selected"].items():
                    _append(sel_path, {
                        "space": space, "repeat": repeat, "fold": fold,
                        "feature": feature, "coefficient": coefficient})
                completed += 1

            elapsed = time.time() - started
            rate = elapsed / max(1, completed - len(done))
            print(f"  {space:14s} repeat {repeat + 1}/{n_repeats}  "
                  f"[{completed}/{total}]  ETA "
                  f"{(total - completed) * rate / 60:.0f} min", flush=True)

    return pd.read_csv(fold_path)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean concordance per feature space."""
    ok = folds[folds["status"] == "ok"].copy()
    for column in ("c_test", "c_train", "gap", "n_selected"):
        ok[column] = pd.to_numeric(ok[column], errors="coerce")
    summary = (
        ok.groupby("space")
        .agg(c_test_mean=("c_test", "mean"),
             c_test_sd=("c_test", "std"),
             c_train_mean=("c_train", "mean"),
             gap_mean=("gap", "mean"),
             n_selected_median=("n_selected", "median"),
             n_folds=("c_test", "size"))
        .reset_index()
    )
    summary["Model"] = summary["space"].map(cfg.SPACE_LABELS)
    return summary


def selection_frequency(ds, tag="main") -> pd.DataFrame:
    """How often each feature is retained across folds."""
    path = cfg.CKPT_DIR / f"cv_selected_{tag}.csv"
    if not path.exists():
        return pd.DataFrame()

    sel = pd.read_csv(path)
    folds = pd.read_csv(cfg.CKPT_DIR / f"cv_folds_{tag}.csv")
    folds = folds[folds["status"] == "ok"]

    rows = []
    for space, block in sel.groupby("space"):
        n_folds = int((folds["space"] == space).sum())
        counts = Counter(block["feature"])
        signs = block.groupby("feature")["coefficient"].mean()
        for feature, count in counts.most_common():
            rows.append({
                "space": space,
                "feature": feature,
                "n_selected": count,
                "n_folds": n_folds,
                "frequency": count / max(1, n_folds),
                "mean_coefficient": float(signs[feature]),
                "direction": "higher risk" if signs[feature] > 0 else "lower risk",
            })
    return pd.DataFrame(rows).sort_values(
        ["space", "frequency"], ascending=[True, False])


def optimism_corrected(ds, space, n_boot=None, seed=cfg.SEED):
    """Harrell's bootstrap optimism correction for the C-index."""
    n_boot = cfg.N_BOOTSTRAP_OPTIMISM if n_boot is None else n_boot
    X = ds.X(space)
    binary = ds.binary_indices(space)

    l1_ratio, alpha = _tune(X, ds.event, ds.duration, binary, cfg.SEED)
    if alpha is None:
        return np.nan, np.nan, np.nan

    y_all = Surv.from_arrays(event=ds.event.astype(bool), time=ds.duration)
    fitted = fit_at_alpha(X, y_all, binary, l1_ratio, alpha)
    apparent = c_index(ds.event, ds.duration,
                       risk_scores(fitted, X, alpha=alpha))

    def one_replicate(b):
        rng = np.random.default_rng(seed + b)
        idx = rng.integers(0, len(X), len(X))
        if ds.event[idx].sum() < 15:
            return None
        try:
            l1_b, alpha_b = _tune(
                X[idx], ds.event[idx], ds.duration[idx], binary, cfg.SEED + b)
            if alpha_b is None:
                return None
            y_b = Surv.from_arrays(
                event=ds.event[idx].astype(bool), time=ds.duration[idx])
            pipe = fit_at_alpha(X[idx], y_b, binary, l1_b, alpha_b)
            c_boot = c_index(ds.event[idx], ds.duration[idx],
                             risk_scores(pipe, X[idx], alpha=alpha_b))
            c_orig = c_index(ds.event, ds.duration,
                             risk_scores(pipe, X, alpha=alpha_b))
            return c_boot - c_orig
        except Exception:
            return None

    values = Parallel(n_jobs=cfg.N_JOBS)(
        delayed(one_replicate)(b) for b in range(n_boot))
    values = [v for v in values if v is not None]
    optimism = float(np.mean(values)) if values else 0.0

    return apparent, optimism, apparent - optimism


def final_model(ds, space, tag="main"):
    """Refit on the whole cohort using the modal penalty from the outer folds."""
    path = cfg.CKPT_DIR / f"cv_folds_{tag}.csv"
    l1_ratio, alpha = None, None

    if path.exists():
        folds = pd.read_csv(path)
        block = folds[(folds["space"] == space) & (folds["status"] == "ok")]
        if not block.empty:
            l1_ratio = float(block["l1_ratio"].median())
            l1_ratio = min(cfg.L1_RATIOS,
                           key=lambda v: abs(v - l1_ratio))
            alpha = float(block["alpha"].median())
            print(f"  modal penalty from folds: l1_ratio={l1_ratio}, "
                  f"alpha={alpha:.5f}")

    X = ds.X(space)
    binary = ds.binary_indices(space)
    if alpha is None:
        print("  WARNING: no fold penalties found; tuning on the full cohort")
        l1_ratio, alpha = _tune(X, ds.event, ds.duration, binary, cfg.SEED)

    y_all = Surv.from_arrays(event=ds.event.astype(bool), time=ds.duration)
    pipeline = fit_at_alpha(X, y_all, binary, l1_ratio, alpha)

    coefficients = selected_features(pipeline, ds.columns_for(space), alpha=alpha)
    score = risk_scores(pipeline, X, alpha=alpha)

    return {
        "space": space,
        "pipeline": pipeline,
        "l1_ratio": l1_ratio,
        "alpha": alpha,
        "coefficients": coefficients,
        "score": score,
        "c_index": c_index(ds.event, ds.duration, score),
    }


def rad_score_table(model: dict) -> pd.DataFrame:
    """Coefficients of the published radiomic score, as hazard ratios."""
    rows = [
        {"feature": name,
         "coefficient": coefficient,
         "hazard_ratio": float(np.exp(coefficient)),
         "direction": "higher risk" if coefficient > 0 else "lower risk"}
        for name, coefficient in model["coefficients"].items()
    ]
    return pd.DataFrame(rows).sort_values(
        "coefficient", key=abs, ascending=False).reset_index(drop=True)


def nadeau_bengio(a, b, n_test, n_train) -> dict:
    """Corrected resampled t-test."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    if k < 2 or d.var(ddof=1) == 0:
        return {"mean_difference": float(d.mean()), "t": np.nan, "p": np.nan}
    correction = 1.0 / k + n_test / n_train
    t = d.mean() / np.sqrt(correction * d.var(ddof=1))
    return {"mean_difference": float(d.mean()), "t": float(t),
            "p": float(2 * (1 - stats.t.cdf(abs(t), df=k - 1)))}


def compare_spaces(ds, folds: pd.DataFrame, spaces=None) -> pd.DataFrame:
    """Pairwise comparison of feature spaces on fold-level concordance."""
    ok = folds[folds["status"] == "ok"]
    spaces = spaces or [s for s in cfg.FEATURE_SPACES
                        if s in ok["space"].unique()]
    n = len(ds.df)
    n_test = n / cfg.N_SPLITS_OUTER

    by_space = {
        space: block.sort_values(["repeat", "fold"])["c_test"].to_numpy()
        for space, block in ok[ok["space"].isin(spaces)].groupby("space")
    }

    mixed = next((s for s in spaces if s.startswith("mixed")), None)
    radiomic = next((s for s in spaces if s.startswith("radiomic")), None)
    pairs = [(mixed, "clinical"), (mixed, radiomic), (radiomic, "clinical")]

    rows = []
    for a, b in [p for p in pairs if None not in p and p[0] != p[1]]:
        if a not in by_space or b not in by_space:
            continue
        length = min(len(by_space[a]), len(by_space[b]))
        result = nadeau_bengio(by_space[a][:length], by_space[b][:length],
                               n_test, n - n_test)
        rows.append({
            "comparison": f"{cfg.SPACE_LABELS.get(a, a)} vs "
                          f"{cfg.SPACE_LABELS.get(b, b)}",
            "space_a": a, "space_b": b, **result,
        })
    return pd.DataFrame(rows)


def compare_all_spaces(ds, folds: pd.DataFrame) -> pd.DataFrame:
    """Pairwise comparison of every evaluated feature space."""
    from itertools import combinations

    ok = folds[folds["status"] == "ok"]
    n = len(ds.df)
    n_test = n / cfg.N_SPLITS_OUTER

    by_space = {
        space: block.sort_values(["repeat", "fold"])["c_test"].to_numpy()
        for space, block in ok.groupby("space")
    }

    rows = []
    for a, b in combinations(sorted(by_space), 2):
        length = min(len(by_space[a]), len(by_space[b]))
        result = nadeau_bengio(by_space[a][:length], by_space[b][:length],
                               n_test, n - n_test)
        rows.append({
            "comparison": f"{cfg.SPACE_LABELS.get(a, a)} vs "
                          f"{cfg.SPACE_LABELS.get(b, b)}",
            "space_a": a, "space_b": b, **result,
        })
    return pd.DataFrame(rows).sort_values("p")


def likelihood_ratio_rad_score(ds, rad_score: np.ndarray) -> dict:
    """Does the radiomic score add to the clinical model, and vice versa?"""
    from lifelines import CoxPHFitter

    base = ds.df[ds.clinical_features].copy()
    base["duration"] = ds.duration
    base["event"] = ds.event

    combined = base.copy()
    combined["rad_score"] = rad_score

    only_score = pd.DataFrame({
        "rad_score": rad_score,
        "duration": ds.duration,
        "event": ds.event,
    })

    fit = lambda frame: CoxPHFitter().fit(
        frame, duration_col="duration", event_col="event")

    cox_clinical = fit(base)
    cox_score = fit(only_score)
    cox_both = fit(combined)

    from scipy.stats import chi2

    lr_score_added = 2 * (cox_both.log_likelihood_ - cox_clinical.log_likelihood_)
    lr_clinical_added = 2 * (cox_both.log_likelihood_ - cox_score.log_likelihood_)

    return {
        "c_clinical": cox_clinical.concordance_index_,
        "c_rad_score": cox_score.concordance_index_,
        "c_combined": cox_both.concordance_index_,
        "lr_score_added": float(lr_score_added),
        "p_score_added": float(chi2.sf(lr_score_added, 1)),
        "lr_clinical_added": float(lr_clinical_added),
        "p_clinical_added": float(
            chi2.sf(lr_clinical_added, len(ds.clinical_features))),
        "model_combined": cox_both,
    }
