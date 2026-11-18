"""Robustness checks: proportional hazards, follow-up windows, crossover, time interaction, subcohorts, sample size, Monte Carlo and decision curve."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

import config_cox as cfg
from evaluation_cox import final_model

warnings.filterwarnings("ignore")


def build_frame(ds, rad_score: np.ndarray) -> pd.DataFrame:
    """Clinical variables plus the radiomic score, in one table."""
    frame = ds.df[list(ds.clinical_features)].copy()
    frame["rad_score"] = rad_score
    frame["duration"] = ds.duration
    frame["event"] = ds.event
    return frame.reset_index(drop=True)


def oof_scores(ds, space: str, tag: str = "main") -> np.ndarray:
    """Out-of-fold risk score for every patient, averaged over the repeats."""
    path = cfg.CKPT_DIR / f"cv_risk_{tag}.csv"
    risk = pd.read_csv(path)
    block = risk[risk["space"] == space]
    if block.empty:
        raise ValueError(f"No out-of-fold predictions for {space!r}; run the "
                         f"cv stage for that space first.")
    return (block.groupby("patient_index")["risk"].mean()
            .reindex(range(len(ds.df))).to_numpy())


def _cindex(duration, event, score) -> float:
    from sksurv.metrics import concordance_index_censored
    return float(concordance_index_censored(
        np.asarray(event).astype(bool), np.asarray(duration, float),
        np.asarray(score, float))[0])


def cv_concordance(frame: pd.DataFrame, columns, repeats=None, penalizer=0.05):
    """Repeated cross-validated C-index for a plain Cox model."""
    repeats = cfg.N_REPEATS if repeats is None else repeats
    scores = []
    for r in range(repeats):
        splitter = KFold(cfg.N_SPLITS_OUTER, shuffle=True,
                         random_state=cfg.SEED + r)
        for train, test in splitter.split(frame):
            try:
                model = CoxPHFitter(penalizer=penalizer).fit(
                    frame.iloc[train][list(columns) + ["duration", "event"]],
                    duration_col="duration", event_col="event")
                risk = model.predict_partial_hazard(frame.iloc[test][list(columns)])
                scores.append(concordance_index(
                    frame.duration.iloc[test], -risk, frame.event.iloc[test]))
            except Exception:
                continue
    if not scores:
        return np.nan, np.nan
    return float(np.mean(scores)), float(np.std(scores))


def proportional_hazards(frame: pd.DataFrame) -> pd.DataFrame:
    """Scaled Schoenfeld residuals for every covariate."""
    columns = [c for c in frame.columns if c not in ("duration", "event")]
    model = CoxPHFitter(penalizer=0.01).fit(
        frame, duration_col="duration", event_col="event")
    test = proportional_hazard_test(model, frame, time_transform="rank")

    table = test.summary[["test_statistic", "p"]].copy()
    table.index.name = "variable"
    table = table.reset_index()
    table["assumption"] = np.where(table["p"] < 0.05, "violated", "satisfied")
    return table.sort_values("p")


def time_varying_effect(frame: pd.DataFrame, variable: str,
                        split: float = 12.0) -> pd.DataFrame:
    """Estimate one covariate's effect separately before and after a cut-point."""
    columns = [c for c in frame.columns if c not in ("duration", "event")]
    rows = []
    for low, high, label in ((0.0, split, f"0-{split:.0f} months"),
                             (split, np.inf, f">{split:.0f} months")):
        window = frame.copy()
        window["event"] = np.where(
            (window.duration >= low) & (window.duration < high)
            & window.event.astype(bool), 1, 0)
        window["duration"] = np.minimum(window.duration, high) - low
        window = window[window.duration > 0]
        if window.event.sum() < 10:
            continue
        model = CoxPHFitter(penalizer=0.01).fit(
            window, duration_col="duration", event_col="event")
        row = model.summary.loc[variable]
        rows.append({
            "variable": variable,
            "window": label,
            "hazard_ratio": round(float(row["exp(coef)"]), 4),
            "ci_lower": round(float(row["exp(coef) lower 95%"]), 4),
            "ci_upper": round(float(row["exp(coef) upper 95%"]), 4),
            "p": float(row["p"]),
            "events": int(window.event.sum()),
        })
    return pd.DataFrame(rows)


def stratified_model(frame: pd.DataFrame, by: str) -> dict:
    """Cox stratified by a covariate whose effect varies over time."""
    standard = CoxPHFitter(penalizer=0.01).fit(
        frame, duration_col="duration", event_col="event")
    stratified = CoxPHFitter(penalizer=0.01).fit(
        frame, duration_col="duration", event_col="event", strata=[by])

    summary = stratified.summary[["exp(coef)", "exp(coef) lower 95%",
                                  "exp(coef) upper 95%", "p"]].round(4)
    summary.columns = ["HR", "CI lower", "CI upper", "p"]
    return {
        "c_standard": float(standard.concordance_index_),
        "c_stratified": float(stratified.concordance_index_),
        "summary": summary.reset_index().rename(columns={"covariate": "variable"}),
    }


def temporal_drift(ds, rad_score: np.ndarray) -> pd.DataFrame:
    """Does the date of diagnosis predict survival, or the radiomic score?"""
    from scipy.stats import spearmanr

    years = ((ds.df["date_diagnosis"] - ds.df["date_diagnosis"].min())
             .dt.days / 365.25).to_numpy()

    frame = pd.DataFrame({"years_since_start": years,
                          "duration": ds.duration, "event": ds.event})
    model = CoxPHFitter().fit(frame, duration_col="duration", event_col="event")
    row = model.summary.loc["years_since_start"]
    rho, p_rho = spearmanr(years, rad_score)

    return pd.DataFrame([
        {"check": "Date of diagnosis as a predictor of survival",
         "estimate": f"HR {row['exp(coef)']:.3f} per year",
         "p": float(row["p"]),
         "interpretation": "drift" if row["p"] < 0.05 else "no drift"},
        {"check": "Correlation between date and radiomic score",
         "estimate": f"rho = {rho:+.3f}", "p": float(p_rho),
         "interpretation": "drift" if p_rho < 0.05 else "no drift"},
    ])


def field_subcohorts(ds, frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Performance within each scanner separately."""
    from data_cox import field_strength

    field = field_strength(ds.df[cfg.ID_COL])
    rows = []
    for value, label in ((1.5, "1.5T only"), (3.0, "3T only"),
                         (None, "Whole cohort")):
        mask = np.ones(len(frame), bool) if value is None else field == value
        if mask.sum() < 25:
            continue
        mean, sd = cv_concordance(frame[mask].reset_index(drop=True), columns)
        rows.append({"subcohort": label, "n": int(mask.sum()),
                     "events": int(ds.event[mask].sum()),
                     "c_index": round(mean, 4), "sd": round(sd, 4)})
    return pd.DataFrame(rows)


def biopsy_sensitivity(ds, frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Performance excluding patients who had only a diagnostic biopsy."""
    from data_cox import _read_clinical_csv, _read_ods

    path = cfg.DATA_CLINICAL
    raw = (_read_ods(path) if str(path).lower().endswith(".ods")
           else _read_clinical_csv(path))
    raw["_id"] = pd.to_numeric(raw[cfg.ID_COL], errors="coerce")
    kind = (raw.set_index("_id")["RESEZIONE"].fillna("").astype(str)
            .str.strip().str.upper())
    mapped = ds.df[cfg.ID_COL].map(kind).fillna("")

    biopsy = (mapped == "BIOPSIA").to_numpy()
    rows = []
    for mask, label in ((~biopsy, "Excluding biopsy-only"),
                        (np.ones(len(frame), bool), "Whole cohort")):
        mean, sd = cv_concordance(frame[mask].reset_index(drop=True), columns)
        rows.append({"cohort": label, "n": int(mask.sum()),
                     "events": int(ds.event[mask].sum()),
                     "c_index": round(mean, 4), "sd": round(sd, 4)})
    table = pd.DataFrame(rows)
    table.attrs["counts"] = mapped.value_counts().to_dict()
    return table


def landmark(ds, spaces, tag="main", horizons=(6.0, 12.0)) -> pd.DataFrame:
    """Discrimination in an early and in a late window of follow-up."""
    rows = []

    def add(space, window, order, label, n, events, value):
        rows.append({"space": space, "Model": cfg.SPACE_LABELS.get(space, space),
                     "window": window, "order": order, "population": label,
                     "n": int(n), "events": int(events),
                     "c_index": round(float(value), 4)})

    for space in spaces:
        score = oof_scores(ds, space, tag)
        d, e = ds.duration, ds.event

        for k, t in enumerate(sorted(horizons)):
            truncated_event = (d < t) & (e == 1)
            add(space, "early", k, f"First {t:.0f} months", len(d),
                truncated_event.sum(),
                _cindex(np.minimum(d, t), truncated_event, score))

        add(space, "whole", len(horizons), "Whole follow-up", len(d), e.sum(),
            _cindex(d, e, score))

        for k, t in enumerate(sorted(horizons)):
            alive = d >= t
            if alive.sum() < 15 or e[alive].sum() < 10:
                continue
            add(space, "late", len(horizons) + 1 + k, f"Alive at {t:.0f} months",
                alive.sum(), e[alive].sum(),
                _cindex(d[alive] - t, e[alive], score[alive]))

    return pd.DataFrame(rows)


def _safe_cindex(duration, event, score) -> float:
    """C-index, or NaN when the window holds no comparable pair."""
    try:
        if np.asarray(event).sum() < 1:
            return np.nan
        return _cindex(duration, event, score)
    except Exception:
        return np.nan


def _window_cindices(ds, scores: dict, idx, horizon: float) -> dict:
    """C-index of each score in the early window and among survivors."""
    d, e = ds.duration[idx], ds.event[idx]
    early_event = (d < horizon) & (e == 1)
    alive = d >= horizon
    out = {}
    for space, score in scores.items():
        s = score[idx]
        out[(space, "early")] = _safe_cindex(np.minimum(d, horizon),
                                             early_event, s)
        out[(space, "late")] = (
            _safe_cindex(d[alive] - horizon, e[alive], s[alive])
            if alive.sum() > 5 else np.nan)
    return out


def crossover_test(ds, spaces, tag="main", horizons=(6.0, 12.0),
                   n_boot=2000, seed=11) -> pd.DataFrame:
    """Test whether the radiomic-minus-clinical advantage reverses between early deaths and survivors."""
    radiomic = next(s for s in spaces if s.startswith("radiomic"))
    reference = "clinical"
    scores = {s: oof_scores(ds, s, tag) for s in (radiomic, reference)}
    everyone = np.arange(len(ds.duration))
    rng = np.random.default_rng(seed)

    def advantage(idx, horizon):
        w = _window_cindices(ds, scores, idx, horizon)
        early = w[(radiomic, "early")] - w[(reference, "early")]
        late = w[(radiomic, "late")] - w[(reference, "late")]
        return early, late

    rows = []
    for horizon in horizons:
        early, late = advantage(everyone, horizon)
        boot_early, boot_late = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(everyone), len(everyone))
            e_b, l_b = advantage(idx, horizon)
            if not (np.isnan(e_b) or np.isnan(l_b)):
                boot_early.append(e_b)
                boot_late.append(l_b)
        boot_early, boot_late = np.asarray(boot_early), np.asarray(boot_late)
        boot = boot_early - boot_late
        lo, hi = np.percentile(boot, [2.5, 97.5])
        e_lo, e_hi = np.percentile(boot_early, [2.5, 97.5])
        l_lo, l_hi = np.percentile(boot_late, [2.5, 97.5])
        p = max(2 * min((boot <= 0).mean(), (boot >= 0).mean()),
                1.0 / (len(boot) + 1))
        alive = ds.duration >= horizon
        rows.append({
            "horizon_months": horizon,
            "deaths_early_window": int(((ds.duration < horizon)
                                        & (ds.event == 1)).sum()),
            "deaths_survivors": int(ds.event[alive].sum()),
            "advantage_early": round(float(early), 4),
            "early_ci_lower": round(float(e_lo), 4),
            "early_ci_upper": round(float(e_hi), 4),
            "advantage_survivors": round(float(late), 4),
            "survivors_ci_lower": round(float(l_lo), 4),
            "survivors_ci_upper": round(float(l_hi), 4),
            "crossover": round(float(early - late), 4),
            "ci_lower": round(float(lo), 4), "ci_upper": round(float(hi), 4),
            "bootstrap_p": round(float(p), 4), "n_boot": len(boot),
            "bonferroni_alpha": round(0.05 / len(horizons), 4),
            "note": "exploratory; scores treated as fixed; hypothesis "
                    "formulated after the pattern was seen",
        })
    return pd.DataFrame(rows)


def window_intervals(ds, spaces, tag="main", horizons=(6.0, 12.0),
                     n_boot=2000, seed=11) -> pd.DataFrame:
    """C-index of each model in the early window and among survivors, with bootstrap 95% CIs."""
    scores = {s: oof_scores(ds, s, tag) for s in spaces}
    everyone = np.arange(len(ds.duration))
    rng = np.random.default_rng(seed)
    rows = []
    for horizon in horizons:
        observed = _window_cindices(ds, scores, everyone, horizon)
        draws = {key: [] for key in observed}
        for _ in range(n_boot):
            idx = rng.integers(0, len(everyone), len(everyone))
            for key, value in _window_cindices(ds, scores, idx, horizon).items():
                if not np.isnan(value):
                    draws[key].append(value)
        early_deaths = int(((ds.duration < horizon) & (ds.event == 1)).sum())
        alive = ds.duration >= horizon
        for (space, window), value in observed.items():
            lo, hi = np.percentile(draws[(space, window)], [2.5, 97.5])
            rows.append({
                "horizon_months": horizon, "space": space,
                "Model": cfg.SPACE_LABELS.get(space, space),
                "window": window, "c_index": round(float(value), 4),
                "ci_lower": round(float(lo), 4), "ci_upper": round(float(hi), 4),
                "n": int(len(everyone) if window == "early" else alive.sum()),
                "deaths": early_deaths if window == "early"
                          else int(ds.event[alive].sum()),
            })
    return pd.DataFrame(rows)


def crossover_power(ds, spaces, tag="main", horizons=(6.0, 12.0),
                    sizes=(104, 150, 200, 300, 400, 600), n_draws=300,
                    seed=5) -> pd.DataFrame:
    """Power to detect the observed differences at larger cohort sizes."""
    from scipy.stats import norm

    radiomic = next(s for s in spaces if s.startswith("radiomic"))
    scores = {s: oof_scores(ds, s, tag) for s in (radiomic, "clinical")}
    everyone = np.arange(len(ds.duration))
    rng = np.random.default_rng(seed)

    def advantage(idx, horizon):
        w = _window_cindices(ds, scores, idx, horizon)
        return (w[(radiomic, "early")] - w[("clinical", "early")],
                w[(radiomic, "late")] - w[("clinical", "late")])

    def power(estimate, se):
        z = abs(estimate) / se
        return float(norm.cdf(z - 1.96) + norm.cdf(-z - 1.96))

    rows = []
    for horizon in horizons:
        obs_early, obs_late = advantage(everyone, horizon)
        for n in sizes:
            early, late, deaths = [], [], []
            for _ in range(n_draws):
                idx = rng.integers(0, len(everyone), n)
                e, l = advantage(idx, horizon)
                if not (np.isnan(e) or np.isnan(l)):
                    early.append(e)
                    late.append(l)
                    deaths.append(int(((ds.duration[idx] < horizon)
                                       & (ds.event[idx] == 1)).sum()))
            early, late = np.asarray(early), np.asarray(late)
            cross = early - late
            rows.append({
                "horizon_months": horizon, "n": n,
                "expected_early_deaths": round(float(np.mean(deaths))),
                "se_early": round(float(early.std()), 4),
                "power_early": round(power(obs_early, early.std()), 3),
                "se_survivors": round(float(late.std()), 4),
                "power_survivors": round(power(obs_late, late.std()), 3),
                "se_crossover": round(float(cross.std()), 4),
                "power_crossover": round(power(obs_early - obs_late,
                                               cross.std()), 3),
                "observed_early": round(float(obs_early), 4),
                "observed_survivors": round(float(obs_late), 4),
                "note": "assumes the observed effect is the true one; "
                        "small-sample effects tend to be overestimated",
            })
    return pd.DataFrame(rows)


def crossover_power(ds, spaces, tag="main", horizons=(6.0, 12.0),
                    sizes=(104, 150, 200, 300, 400, 600), n_draws=300,
                    seed=5) -> pd.DataFrame:
    """Cohort size needed for the crossover analysis to exclude zero."""
    from scipy.stats import norm

    radiomic = next(s for s in spaces if s.startswith("radiomic"))
    reference = "clinical"
    scores = {s: oof_scores(ds, s, tag) for s in (radiomic, reference)}
    rng = np.random.default_rng(seed)

    def advantage(idx, horizon):
        w = _window_cindices(ds, scores, idx, horizon)
        return (w[(radiomic, "early")] - w[(reference, "early")],
                w[(radiomic, "late")] - w[(reference, "late")])

    def power(estimate, se):
        z = abs(estimate) / se
        return float(norm.cdf(z - 1.96) + norm.cdf(-z - 1.96))

    rows = []
    for horizon in horizons:
        obs_early, obs_late = advantage(np.arange(len(ds.duration)), horizon)
        obs_cross = obs_early - obs_late
        for n in sizes:
            early, late, early_deaths = [], [], []
            for _ in range(n_draws):
                idx = rng.integers(0, len(ds.duration), n)
                e, l = advantage(idx, horizon)
                if np.isnan(e) or np.isnan(l):
                    continue
                early.append(e)
                late.append(l)
                early_deaths.append(int(((ds.duration[idx] < horizon)
                                         & (ds.event[idx] == 1)).sum()))
            early, late = np.asarray(early), np.asarray(late)
            cross = early - late
            rows.append({
                "horizon_months": horizon, "n": n,
                "expected_early_deaths": round(float(np.mean(early_deaths))),
                "observed_advantage_early": round(float(obs_early), 4),
                "power_early": round(power(obs_early, early.std()), 3),
                "observed_advantage_survivors": round(float(obs_late), 4),
                "power_survivors": round(power(obs_late, late.std()), 3),
                "observed_crossover": round(float(obs_cross), 4),
                "power_crossover": round(power(obs_cross, cross.std()), 3),
                "note": "projection assuming the observed effect is true; "
                        "required sizes are more likely underestimated",
            })
    return pd.DataFrame(rows)


def _hazard_ratio_by_window(ds, X: pd.DataFrame, focus: str,
                            split: float) -> dict:
    """Hazard ratio of `focus` before and after `split`, with the interaction p-value."""
    from lifelines import CoxTimeVaryingFitter

    d, e = ds.duration, ds.event
    columns = list(X.columns)
    values = X.to_numpy(float)
    rows = []
    for i in range(len(d)):
        if d[i] <= split:
            rows.append((i, 0.0, d[i], e[i], 0, *values[i]))
        else:
            rows.append((i, 0.0, split, 0, 0, *values[i]))
            rows.append((i, split, d[i], e[i], 1, *values[i]))
    frame = pd.DataFrame(rows, columns=["id", "start", "stop", "event",
                                        "late"] + columns)
    frame["interaction"] = frame[focus] * frame["late"]
    keys = dict(id_col="id", start_col="start", stop_col="stop",
                event_col="event")

    joint = CoxTimeVaryingFitter().fit(
        frame[["id", "start", "stop", "event"] + columns + ["interaction"]],
        **keys).summary

    result = {"ratio": float(np.exp(joint.loc["interaction", "coef"])),
              "p_interaction": float(joint.loc["interaction", "p"])}
    for label, sub in (("early", frame[frame.late == 0]),
                       ("late", frame[frame.late == 1])):
        fit = CoxTimeVaryingFitter().fit(
            sub[["id", "start", "stop", "event"] + columns], **keys).summary
        r = fit.loc[focus]
        result[label] = (float(np.exp(r["coef"])),
                         float(np.exp(r["coef lower 95%"])),
                         float(np.exp(r["coef upper 95%"])),
                         int(sub.event.sum()))
    return result


def time_interaction(ds, tag="main", split=12.0) -> pd.DataFrame:
    """Does the hazard ratio of each score, or of MGMT, change after `split`?"""
    spaces = list(cfg.FEATURE_SPACES)
    rad = next(s for s in spaces if s.startswith("radiomic"))
    mix = next(s for s in spaces if s.startswith("mixed"))

    def z(v):
        return (v - v.mean()) / v.std()

    entries = []
    for label, space in (("Radiomic score", rad), ("Clinical score", "clinical"),
                         ("Mixed score", mix)):
        entries.append((label, pd.DataFrame(
            {"x": z(oof_scores(ds, space, tag))}), "x"))
    entries.append(("MGMT (unadjusted)", pd.DataFrame(
        {"x": ds.df["MGMT_METILAZIONE"].to_numpy(float)}), "x"))

    adjusted = ds.df[list(ds.clinical_features)].reset_index(drop=True).copy()
    adjusted["rad_score"] = z(oof_scores(ds, rad, tag))
    entries.append(("MGMT (adjusted)", adjusted, "MGMT_METILAZIONE"))

    rows = []
    for label, X, focus in entries:
        r = _hazard_ratio_by_window(ds, X, focus, split)
        rows.append({
            "term": label,
            f"hr_0_{split:.0f}m": round(r["early"][0], 3),
            "ci_early": f"{r['early'][1]:.2f}-{r['early'][2]:.2f}",
            "events_early": r["early"][3],
            f"hr_after_{split:.0f}m": round(r["late"][0], 3),
            "ci_late": f"{r['late'][1]:.2f}-{r['late'][2]:.2f}",
            "events_late": r["late"][3],
            "ratio_late_over_early": round(r["ratio"], 3),
            "p_interaction": round(r["p_interaction"], 4),
        })
    return pd.DataFrame(rows)


def pair_types(ds, spaces, tag="main", horizon=12.0) -> pd.DataFrame:
    """Concordance by type of patient pair, out-of-fold, derivation cohort."""
    from validate_external import pair_decomposition

    rows = []
    for space in spaces:
        result = pair_decomposition(ds.duration, ds.event,
                                    oof_scores(ds, space, tag), horizon)
        for kind, value in result.items():
            rows.append({"space": space,
                         "Model": cfg.SPACE_LABELS.get(space, space),
                         "pair_type": kind, "c_index": value["c_index"],
                         "pairs": value["pairs"]})
    return pd.DataFrame(rows)


def sample_size(ds, spaces, tag="main", sizes=(40, 60, 80, 100, 150, 200, 300),
                n_draws=400, seed=1) -> pd.DataFrame:
    """Precision of the C-index at different validation-cohort sizes."""
    scores = {s: oof_scores(ds, s, tag) for s in spaces}
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        values = {s: [] for s in spaces}
        events = []
        for _ in range(n_draws):
            idx = rng.integers(0, len(ds.duration), n)
            if ds.event[idx].sum() < 5:
                continue
            events.append(ds.event[idx].sum())
            for s in spaces:
                values[s].append(_cindex(ds.duration[idx], ds.event[idx],
                                         scores[s][idx]))
        for s in spaces:
            lo, hi = np.percentile(values[s], [2.5, 97.5])
            rows.append({"space": s, "Model": cfg.SPACE_LABELS.get(s, s),
                         "n": n, "expected_events": round(float(np.mean(events))),
                         "ci_width": round(float(hi - lo), 4)})
    return pd.DataFrame(rows)


def monte_carlo(ds, spaces, n_splits=150, test_size=0.2, seed=2026) -> pd.DataFrame:
    """Paired random train/test splits, each model refitted with its own tuning."""
    import os

    from joblib import Parallel, delayed
    from sklearn.model_selection import ShuffleSplit
    from sksurv.util import Surv

    from evaluation_cox import _tune, c_index
    from pipelines_cox import fit_at_alpha, risk_scores

    path = cfg.CKPT_DIR / "monte_carlo.csv"
    splits = list(ShuffleSplit(n_splits=n_splits, test_size=test_size,
                               random_state=seed).split(ds.X(spaces[0])))
    done = set(pd.read_csv(path)["split"]) if path.exists() else set()

    def one(i, train, test):
        row = {"split": i}
        for space in spaces:
            X, binary = ds.X(space), ds.binary_indices(space)
            try:
                l1, alpha = _tune(X[train], ds.event[train], ds.duration[train],
                                  binary, seed + i)
                y = Surv.from_arrays(event=ds.event[train].astype(bool),
                                     time=ds.duration[train])
                model = fit_at_alpha(X[train], y, binary, l1, alpha)
                row[space] = c_index(ds.event[test], ds.duration[test],
                                     risk_scores(model, X[test], alpha=alpha))
            except Exception:
                row[space] = np.nan
        return row

    pending = [(i, tr, te) for i, (tr, te) in enumerate(splits) if i not in done]
    for start in range(0, len(pending), 20):
        batch = pending[start:start + 20]
        rows = Parallel(n_jobs=cfg.N_JOBS)(delayed(one)(*b) for b in batch)
        pd.DataFrame(rows).to_csv(path, mode="a", index=False,
                                  header=not path.exists())
        print(f"    {len(done) + start + len(batch)}/{n_splits} splits", flush=True)

    return pd.read_csv(path).dropna()


def monte_carlo_summary(results: pd.DataFrame, spaces) -> pd.DataFrame:
    rows = []
    for space in spaces:
        v = results[space]
        rows.append({"comparison": cfg.SPACE_LABELS.get(space, space),
                     "median": round(v.median(), 4),
                     "p2.5": round(v.quantile(.025), 4),
                     "p97.5": round(v.quantile(.975), 4), "share": np.nan})
    mixed = next((s for s in spaces if s.startswith("mixed")), None)
    for other in [s for s in spaces if s != mixed]:
        diff = results[mixed] - results[other]
        rows.append({"comparison": f"{cfg.SPACE_LABELS.get(mixed)} > "
                                   f"{cfg.SPACE_LABELS.get(other)}",
                     "median": round(diff.median(), 4),
                     "p2.5": round(diff.quantile(.025), 4),
                     "p97.5": round(diff.quantile(.975), 4),
                     "share": round(float((diff > 0).mean()), 4)})
    return pd.DataFrame(rows)


def reduced_model(ds, space="mixed_shape", tag="main") -> pd.DataFrame:
    """Does trimming the smallest components change anything?"""
    model = final_model(ds, space, tag=tag)
    magnitude = pd.Series(model["coefficients"]).abs().sort_values(ascending=False)

    columns = ds.columns_for(space)
    X = pd.DataFrame(ds.X(space), columns=columns)
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
    X["duration"] = ds.duration
    X["event"] = ds.event

    rows = []
    for threshold, label in ((0.0, "all non-zero"), (0.02, "|beta| >= 0.02"),
                             (0.05, "|beta| >= 0.05"), (0.10, "|beta| >= 0.10")):
        keep = [c for c in magnitude[magnitude >= threshold].index
                if c in X.columns]
        if len(keep) < 2:
            continue
        mean, sd = cv_concordance(X, keep)
        rows.append({"model": label, "k": len(keep),
                     "c_index": round(mean, 4), "sd": round(sd, 4),
                     "note": "exploratory; features chosen on the full "
                             "cohort, estimate optimistic"})
    return pd.DataFrame(rows)


def decision_curve(ds, frame: pd.DataFrame, clinical, horizon=12.0,
                   thresholds=None, repeats=5) -> pd.DataFrame:
    """Net benefit at a fixed horizon, from out-of-fold predicted risks."""
    thresholds = (np.arange(0.2, 0.85, 0.05) if thresholds is None
                  else np.asarray(thresholds))

    def out_of_fold_risk(columns):
        accumulated = np.full((len(frame), repeats), np.nan)
        for r in range(repeats):
            splitter = KFold(cfg.N_SPLITS_OUTER, shuffle=True,
                             random_state=cfg.SEED + r)
            for train, test in splitter.split(frame):
                try:
                    model = CoxPHFitter(penalizer=0.05).fit(
                        frame.iloc[train][list(columns) + ["duration", "event"]],
                        duration_col="duration", event_col="event")
                    survival = model.predict_survival_function(
                        frame.iloc[test][list(columns)], times=[horizon])
                    accumulated[test, r] = 1 - survival.values[0]
                except Exception:
                    continue
        return np.nanmean(accumulated, axis=1)

    risks = {
        "Clinical": out_of_fold_risk(clinical),
        "Radiomic score": out_of_fold_risk(["rad_score"]),
        "Combined": out_of_fold_risk(list(clinical) + ["rad_score"]),
    }

    overall = KaplanMeierFitter().fit(ds.duration, ds.event)
    prevalence = 1 - float(overall.predict(horizon))

    def net_benefit(probabilities, threshold):
        treated = probabilities >= threshold
        if treated.sum() == 0:
            return 0.0
        km = KaplanMeierFitter().fit(ds.duration[treated], ds.event[treated])
        event_rate = 1 - float(km.predict(horizon))
        true_positive = event_rate * treated.sum()
        false_positive = (1 - event_rate) * treated.sum()
        n = len(probabilities)
        return (true_positive / n
                - (false_positive / n) * (threshold / (1 - threshold)))

    rows = []
    for threshold in thresholds:
        row = {"threshold": round(float(threshold), 3),
               "treat_all": round(prevalence - (1 - prevalence)
                                  * (threshold / (1 - threshold)), 4),
               "treat_none": 0.0}
        for name, probabilities in risks.items():
            row[name] = round(net_benefit(probabilities, threshold), 4)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.attrs["prevalence"] = prevalence
    table.attrs["horizon"] = horizon
    return table


def run_all(ds, tag="main") -> dict:
    """Every check, printed and returned for the figures."""
    spaces = list(cfg.FEATURE_SPACES)
    radiomic_space = next((s for s in spaces if s.startswith("radiomic")), None)

    final_score = final_model(ds, radiomic_space, tag=tag)["score"]
    oof_rad = oof_scores(ds, radiomic_space, tag)
    model_frame = build_frame(ds, final_score)
    frame = build_frame(ds, oof_rad)
    columns = list(ds.clinical_features) + ["rad_score"]

    out: dict = {"frame": frame, "columns": columns}

    print("\n--- Proportional hazards (scaled Schoenfeld residuals) ---")
    ph = proportional_hazards(model_frame)
    print(ph.round(4).to_string(index=False))
    out["proportional_hazards"] = ph

    violated = ph.loc[ph["assumption"] == "violated", "variable"].tolist()
    if violated:
        print(f"\n--- Effect over time for {violated[0]} ---")
        tv = time_varying_effect(model_frame, violated[0])
        print(tv.round(4).to_string(index=False))
        out["time_varying"] = tv

        strat = stratified_model(model_frame, violated[0])
        print(f"\n  C-index: {strat['c_standard']:.3f} standard, "
              f"{strat['c_stratified']:.3f} stratified by {violated[0]}")
        out["stratified"] = strat

    print("\n--- Accrual drift ---")
    drift = temporal_drift(ds, oof_rad)
    print(drift.round(4).to_string(index=False))
    out["drift"] = drift

    print("\n--- Scanner subcohorts ---")
    subs = field_subcohorts(ds, frame, columns)
    print(subs.to_string(index=False))
    out["subcohorts"] = subs

    print("\n--- Excluding biopsy-only patients ---")
    try:
        biopsy = biopsy_sensitivity(ds, frame, columns)
        print(biopsy.to_string(index=False))
        print(f"  resection recorded as: {biopsy.attrs.get('counts')}")
        out["biopsy"] = biopsy
    except Exception as exc:
        print(f"  skipped ({type(exc).__name__}: {exc})")

    print("\n--- Trimming the smallest components ---")
    reduced = reduced_model(ds, tag=tag)
    print(reduced.to_string(index=False))
    out["reduced"] = reduced

    print("\n--- Discrimination by window of follow-up ---")
    lm = landmark(ds, spaces, tag)
    order = lm.drop_duplicates("population").sort_values("order")["population"]
    print(lm.pivot(index="Model", columns="population",
                   values="c_index")[list(order)].round(3).to_string())
    out["landmark"] = lm

    print("\n--- Crossover of the radiomic advantage across follow-up ---")
    crossover = crossover_test(ds, spaces, tag)
    print(crossover.drop(columns=["note", "n_boot"]).to_string(index=False))
    out["crossover"] = crossover

    print("\n--- Power to detect the observed crossover, by cohort size ---")
    cp = crossover_power(ds, spaces, tag)
    shown = cp[cp["horizon_months"] == cp["horizon_months"].min()]
    print(shown[["n", "expected_early_deaths", "power_early",
                 "power_survivors", "power_crossover"]].to_string(index=False))
    out["crossover_power"] = cp

    windows = window_intervals(ds, spaces, tag)
    out["crossover_windows"] = windows

    print("\n--- Power of the crossover analysis by cohort size ---")
    cpower = crossover_power(ds, spaces, tag)
    print(cpower.drop(columns=["note"]).to_string(index=False))
    out["crossover_power"] = cpower

    print("\n--- Does the hazard ratio change over time? ---")
    interaction = time_interaction(ds, tag)
    print(interaction.to_string(index=False))
    out["time_interaction"] = interaction

    print("\n--- Concordance by pair type (derivation, out-of-fold) ---")
    pt = pair_types(ds, spaces, tag)
    print(pt.pivot(index="Model", columns="pair_type",
                   values="c_index").round(3).to_string())
    out["pair_types"] = pt

    print("\n--- Precision of an external validation, by cohort size ---")
    ss = sample_size(ds, spaces, tag)
    print(ss.pivot(index="n", columns="Model", values="ci_width")
          .round(3).to_string())
    out["sample_size"] = ss

    print("\n--- Decision curve analysis ---")
    dca = decision_curve(ds, frame, list(ds.clinical_features))
    print(f"  event rate by {dca.attrs['horizon']:.0f} months: "
          f"{dca.attrs['prevalence']:.3f}")
    print(dca[dca.threshold.isin([0.3, 0.4, 0.5, 0.6, 0.7])].to_string(index=False))
    out["dca"] = dca

    return out
