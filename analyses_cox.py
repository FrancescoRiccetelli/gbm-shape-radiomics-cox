"""Permutation test, risk stratification, calibration, SHAP, time-dependent AUC and orthogonality analyses."""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sksurv.util import Surv

import config_cox as cfg
from evaluation_cox import _tune, c_index
from pipelines_cox import fit_at_alpha, risk_scores


def _cv_concordance(ds, space, event, duration, seed=cfg.SEED) -> float:
    """Mean fold concordance for one (possibly permuted) outcome vector."""
    X = ds.X(space)
    binary = ds.binary_indices(space)
    splitter = KFold(n_splits=cfg.N_SPLITS_OUTER, shuffle=True,
                     random_state=cfg.SEED)

    scores = []
    for train, test in splitter.split(X):
        try:
            l1_ratio, alpha = _tune(
                X[train], event[train], duration[train], binary, seed)
            if alpha is None:
                continue
            y_train = Surv.from_arrays(
                event=event[train].astype(bool), time=duration[train])
            pipeline = fit_at_alpha(X[train], y_train, binary, l1_ratio, alpha)
            risk = risk_scores(pipeline, X[test], alpha=alpha)
            scores.append(c_index(event[test], duration[test], risk))
            del pipeline
            gc.collect()
        except Exception:
            continue

    return float(np.mean(scores)) if scores else np.nan


def _one_permutation(ds, space, index) -> dict:
    """Run permutation `index`, returning a record even when it fails."""
    try:
        rng = np.random.default_rng(cfg.SEED + index)
        order = rng.permutation(len(ds.duration))
        score = _cv_concordance(
            ds, space, ds.event[order], ds.duration[order])
        return {"permutation": index, "c_index": score, "status": "ok",
                "error": ""}
    except Exception as exc:
        return {"permutation": index, "c_index": np.nan, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def _write(path: Path, rows: list[dict]) -> None:
    header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        pd.DataFrame(rows).to_csv(fh, header=header, index=False)
        fh.flush()
        os.fsync(fh.fileno())


def permutation_test(ds, space, n_permutations=None, batch_size=10) -> dict:
    """Permutation test for one feature space, resumable at any point."""
    n_permutations = (cfg.N_PERMUTATIONS if n_permutations is None
                      else n_permutations)
    path = cfg.CKPT_DIR / f"permutation_{space}.csv"
    observed_path = cfg.CKPT_DIR / f"permutation_observed_{space}.txt"

    if observed_path.exists():
        observed = float(observed_path.read_text().strip())
        print(f"  observed C-index (checkpoint): {observed:.4f}")
    else:
        print(f"  computing observed C-index for {space}...")
        observed = _cv_concordance(ds, space, ds.event, ds.duration)
        observed_path.write_text(f"{observed}\n")
        print(f"  observed C-index: {observed:.4f}")

    done: set[int] = set()
    if path.exists():
        done = set(pd.read_csv(path)["permutation"].tolist())
        print(f"  resuming: {len(done)}/{n_permutations} on disk")

    pending = [i for i in range(n_permutations) if i not in done]
    started = time.time()

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        rows = Parallel(n_jobs=cfg.N_JOBS)(
            delayed(_one_permutation)(ds, space, i) for i in batch)
        _write(path, rows)
        gc.collect()

        n_done = len(done) + start + len(batch)
        per_permutation = (time.time() - started) / max(1, start + len(batch))
        print(f"    {n_done}/{n_permutations}  "
              f"({per_permutation:.1f} s each, ETA "
              f"{(n_permutations - n_done) * per_permutation / 3600:.1f} h)",
              flush=True)

    results = pd.read_csv(path)
    null = results.loc[results["status"] == "ok", "c_index"].dropna().to_numpy()
    n_failed = int((results["status"] == "failed").sum())

    p_value = (1 + (null >= observed).sum()) / (1 + len(null))
    resolution = 1 / (1 + len(null))

    print(f"  {space}: C = {observed:.4f}, p = {p_value:.4f} "
          f"(n = {len(null)}, finest resolvable p = {resolution:.4f}"
          + (f", {n_failed} failed)" if n_failed else ")"))

    return {"space": space, "observed": observed, "null": null,
            "p_value": float(p_value), "n": int(len(null)),
            "n_failed": n_failed, "resolution": float(resolution)}


def dry_run(ds, space="mixed_shape", n=3) -> float:
    """Time a few permutations before committing to an overnight job."""
    print(f"Dry run: {n} permutations of {space}...")
    started = time.time()
    Parallel(n_jobs=min(cfg.N_JOBS, n))(
        delayed(_one_permutation)(ds, space, 10_000 + i) for i in range(n))
    elapsed = time.time() - started

    per_permutation = elapsed / n * min(cfg.N_JOBS, n)
    hours = per_permutation * cfg.N_PERMUTATIONS / cfg.N_JOBS / 3600
    print(f"  {per_permutation:.1f} s per permutation (single worker)")
    print(f"  {cfg.N_PERMUTATIONS} permutations on {cfg.N_JOBS} workers: "
          f"~{hours:.1f} h per space")
    return per_permutation


def stratify_by_risk(ds, score: np.ndarray, method="median") -> dict:
    """Split patients into risk groups by the radiomic score and compare survival."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    threshold = np.median(score) if method == "median" else method
    high = score > threshold

    curves = {}
    for label, mask, colour in (("High risk", high, "#C44E52"),
                                ("Low risk", ~high, "#4C72B0")):
        kmf = KaplanMeierFitter().fit(
            ds.duration[mask], ds.event[mask], label=label)
        curves[label] = {
            "fitter": kmf,
            "n": int(mask.sum()),
            "events": int(ds.event[mask].sum()),
            "median": float(kmf.median_survival_time_),
            "colour": colour,
        }

    test = logrank_test(ds.duration[high], ds.duration[~high],
                        ds.event[high], ds.event[~high])

    return {"curves": curves, "threshold": float(threshold),
            "logrank_p": float(test.p_value), "high_mask": high}


def calibration_at(ds, model, horizon: float, n_groups: int = 4) -> pd.DataFrame:
    """Predicted versus observed survival at one horizon."""
    from lifelines import KaplanMeierFitter

    score = model["score"]
    groups = pd.qcut(score, n_groups, labels=False, duplicates="drop")

    space = model.get("space", "mixed_shape")
    try:
        surv_fns = model["pipeline"].predict_survival_function(
            ds.X(space), alpha=model["alpha"])
        predicted = np.array([fn(horizon) for fn in surv_fns])
    except Exception as exc:
        print(f"  calibration at {horizon:.0f} months unavailable: "
              f"{type(exc).__name__}: {exc}")
        predicted = np.full(len(score), np.nan)

    rows = []
    for g in np.unique(groups):
        mask = groups == g
        kmf = KaplanMeierFitter().fit(ds.duration[mask], ds.event[mask])
        observed = float(kmf.predict(horizon))
        rows.append({
            "group": int(g),
            "n": int(mask.sum()),
            "predicted": float(np.nanmean(predicted[mask])),
            "observed": observed,
            "se": float(np.sqrt(max(observed * (1 - observed), 1e-6)
                                / mask.sum())),
        })
    return pd.DataFrame(rows)


def orthogonality(ds, tag="main", space=None) -> pd.DataFrame:
    """Does the radiomic score already contain the clinical information?"""
    from scipy.stats import mannwhitneyu, spearmanr

    space = space or next(
        (s for s in cfg.FEATURE_SPACES if s.startswith("radiomic")), None)
    path = cfg.CKPT_DIR / f"cv_risk_{tag}.csv"
    if space is None or not path.exists():
        return pd.DataFrame()

    risk = pd.read_csv(path)
    block = risk[risk.space == space]
    if block.empty:
        return pd.DataFrame()

    score = (block.groupby("patient_index").risk.mean()
             .reindex(range(len(ds.df))).to_numpy())

    rows = []
    for name in ds.clinical_features:
        values = ds.df[name].to_numpy(dtype=float)
        binary = set(np.unique(values[~np.isnan(values)])) <= {0.0, 1.0}
        if binary:
            a, b = score[values == 1], score[values == 0]
            if len(a) < 3 or len(b) < 3:
                continue
            p = mannwhitneyu(a, b).pvalue
            effect = 2 * mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1
            test = "Mann-Whitney"
        else:
            effect, p = spearmanr(score, values)
            test = "Spearman"
        rows.append({
            "variable": name,
            "test": test,
            "effect": round(float(effect), 3),
            "p": float(p),
            "captured_by_radiomics": "yes" if p < 0.05 else "no",
        })
    return pd.DataFrame(rows).sort_values("p")


def time_dependent_auc(ds, tag="main", horizons=None) -> pd.DataFrame:
    """Cumulative/dynamic AUC at fixed horizons, averaged over folds."""
    from sklearn.model_selection import KFold
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    horizons = cfg.TIME_HORIZONS if horizons is None else horizons
    path = cfg.CKPT_DIR / f"cv_risk_{tag}.csv"
    if not path.exists():
        return pd.DataFrame()

    risk = pd.read_csv(path)
    y_all = Surv.from_arrays(event=ds.event.astype(bool), time=ds.duration)
    n = len(ds.df)

    rows = []
    for space, block in risk.groupby("space"):
        collected = {h: [] for h in horizons}
        for repeat, sub in block.groupby("repeat"):
            scores = sub.set_index("patient_index").risk
            splitter = KFold(cfg.N_SPLITS_OUTER, shuffle=True,
                             random_state=cfg.SEED + int(repeat))
            for _, test in splitter.split(np.arange(n)):
                s = scores.reindex(test).to_numpy()
                if np.isnan(s).any():
                    continue
                y_test = Surv.from_arrays(
                    event=ds.event[test].astype(bool), time=ds.duration[test])
                for h in horizons:
                    if ((ds.duration[test] < h).sum() < 3
                            or (ds.duration[test] >= h).sum() < 3):
                        continue
                    try:
                        auc, _ = cumulative_dynamic_auc(y_all, y_test, s, [h])
                        if np.isfinite(auc[0]):
                            collected[h].append(float(auc[0]))
                    except Exception:
                        continue
        for h, values in collected.items():
            if values:
                rows.append({
                    "space": space,
                    "Model": cfg.SPACE_LABELS.get(space, space),
                    "horizon_months": h,
                    "auc_mean": round(float(np.mean(values)), 4),
                    "auc_sd": round(float(np.std(values)), 4),
                    "n_folds": len(values),
                    "at_risk": int((ds.duration >= h).sum()),
                })
    return pd.DataFrame(rows)


def shap_values(ds, space, model):
    """Shapley values for the final Coxnet model."""
    import shap

    pipeline = model["pipeline"]
    X = ds.X(space)
    X_transformed = pipeline[:-1].transform(X)

    names = np.array(ds.columns_for(space))
    names = names[pipeline.named_steps["variance"].get_support()]
    names = names[pipeline.named_steps["correlation"].get_support()]

    coefficients = pipeline.named_steps["coxnet"].coef_
    if coefficients.ndim > 1:
        alphas = pipeline.named_steps["coxnet"].alphas_
        column = int(np.argmin(np.abs(alphas - model["alpha"])))
        coefficients = coefficients[:, column]

    assert len(names) == X_transformed.shape[1], (
        f"{len(names)} names but {X_transformed.shape[1]} columns")

    centred = X_transformed - X_transformed.mean(axis=0)
    values = centred * coefficients

    keep = np.flatnonzero(coefficients)
    return {
        "values": values[:, keep],
        "X": X_transformed[:, keep],
        "names": names[keep].tolist(),
        "coefficients": coefficients[keep],
    }
