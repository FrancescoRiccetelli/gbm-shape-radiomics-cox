"""Freezes the final models and applies them unchanged to an external cohort."""
from __future__ import annotations

import config_cox as cfg

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = cfg.MODEL_PATH
SPACES_TO_FREEZE = ["clinical", "radiomic_shape", "mixed_shape"]


def _expected_c(lp) -> float:
    lp = np.asarray(lp, dtype=float)
    diff = np.abs(lp[:, None] - lp[None, :])[np.triu_indices(len(lp), k=1)]
    return float(np.mean(1.0 / (1.0 + np.exp(-diff))))


def freeze(spaces=None, tag="main") -> dict:
    """Fit each feature space on the derivation cohort and save everything needed to reproduce the score."""
    from data_cox import load_dataset
    from evaluation_cox import final_model
    from lifelines import KaplanMeierFitter

    spaces = spaces or SPACES_TO_FREEZE
    ds = load_dataset()

    frozen = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "derivation_cohort": {
            "n": int(len(ds.df)),
            "events": int(ds.n_events),
            "median_followup_months": None,
            "data_lock": cfg.DATA_LOCK,
        },
        "clinical_features": list(ds.clinical_features),
        "p53_aberrant_categories": list(cfg.P53_ABERRANT_CATEGORIES),
        "deep_pattern": cfg.DEEP_PATTERN,
        "spaces": {},
    }

    kmf = KaplanMeierFitter().fit(ds.duration, 1 - ds.event)
    frozen["derivation_cohort"]["median_followup_months"] = float(
        kmf.median_survival_time_)

    frozen["scanner_correction"] = cfg.SCANNER_CORRECTION
    if cfg.SCANNER_CORRECTION == "combat":
        from data_cox import add_volume_ratios, load_radiomic

        raw = load_radiomic()
        if cfg.ADD_VOLUME_RATIOS:
            raw = add_volume_ratios(raw)
        raw = raw[raw[cfg.ID_COL].isin(ds.df[cfg.ID_COL])]
        region_columns = [c for c in raw.columns
                          if any(c.startswith(f"{p}_") for p in cfg.REGIONS)]
        frozen["combat"] = {
            "columns": region_columns,
            "pooled_mean": [float(v) for v in raw[region_columns].mean()],
            "pooled_sd": [float(v) for v in raw[region_columns].std(ddof=0)],
        }
        print(f"  ComBat reference stored for {len(region_columns)} features")

    frozen["mgmt_binary_threshold"] = cfg.MGMT_BINARY_THRESHOLD

    for space in spaces:
        print(f"\n--- Freezing {space} ---")
        model = final_model(ds, space, tag=tag)
        pipeline = model["pipeline"]
        columns = ds.columns_for(space)

        kept = np.arange(len(columns))
        kept = kept[pipeline.named_steps["variance"].get_support()]
        scaler = pipeline.named_steps["scaler"]
        kept_after_corr = kept[pipeline.named_steps["correlation"].get_support()]

        score = model["score"]
        threshold = float(np.median(score))

        baseline = None
        try:
            fn = pipeline.named_steps["coxnet"].baseline_survival_
            baseline = {
                "times": [float(t) for t in fn.index],
                "survival": [float(v) for v in fn.iloc[:, 0]],
            }
        except Exception:
            pass

        frozen["spaces"][space] = {
            "feature_order": columns,
            "variance_kept": [int(i) for i in
                              np.flatnonzero(
                                  pipeline.named_steps["variance"].get_support())],
            "correlation_kept": [int(i) for i in
                                 pipeline.named_steps["correlation"].to_keep_],
            "scaler_mean": [float(v) for v in getattr(
                scaler.scaler_, "mean_", [])],
            "scaler_scale": [float(v) for v in getattr(
                scaler.scaler_, "scale_", [])],
            "scaler_continuous_indices": [int(i) for i in scaler.continuous_],
            "l1_ratio": float(model["l1_ratio"]),
            "alpha": float(model["alpha"]),
            "coefficients": {k: float(v) for k, v in model["coefficients"].items()},
            "apparent_c_index": float(model["c_index"]),
            "derivation_expected_c": _expected_c(score),
            "risk_threshold": threshold,
            "score_mean": float(np.mean(score)),
            "score_sd": float(np.std(score)),
            "baseline_survival": baseline,
        }
        print(f"  {len(model['coefficients'])} non-zero coefficients, "
              f"apparent C-index {model['c_index']:.3f}")

    MODEL_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"\nFrozen model written to {MODEL_PATH}")
    print("Do not refit, reselect or re-threshold this model on external data.")

    _write_protocol(frozen)
    return frozen


def _write_protocol(frozen: dict) -> None:
    """A plain-language description to send to the validating centre."""
    lines = [
        "# Validation protocol",
        "",
        f"Model frozen on {frozen['created']} from a derivation cohort of "
        f"{frozen['derivation_cohort']['n']} patients "
        f"({frozen['derivation_cohort']['events']} deaths).",
        "",
        "## Data required",
        "",
        "For every patient:",
        "",
        "- date of histological diagnosis",
        "- date of death, or date of last contact if alive",
        "- age at diagnosis, sex",
        "- extent of resection (gross total resection versus anything else)",
        "- Ki67 (%), MGMT methylation (%)",
        "- p53 immunohistochemistry, as one of: assente, rare cellule, "
        "debole, intermedia, intensa, diffusa",
        "- tumour location",
        "",
        "## Definitions",
        "",
        "**Deep involvement.** Recorded as present when the tumour involves "
        "the thalamus, basal ganglia, internal capsule, corpus callosum, "
        "trigone, or mesial structures. This was the only location grouping "
        "that separated survival in the derivation cohort.",
        "",
        "**p53.** Grouped as aberrant (complete absence, rare cells, intense "
        "or diffuse overexpression) versus wild-type pattern (weak or "
        "intermediate). Both extremes of staining indicate an altered gene; "
        "intermediate expression is the wild-type pattern.",
        "",
        "## Imaging",
        "",
        "Segmentation with the same automated tool and settings used in the "
        "derivation cohort, producing enhancing tumour, necrotic core and "
        "peritumoral edema as separate regions.",
        "",
        "## What must not change",
        "",
        "The model is applied exactly as frozen. Coefficients are not "
        "refitted, features are not reselected, and the risk threshold is "
        "not recomputed on the validation data.",
        "",
        "## Frozen parameters",
        "",
    ]
    for space, spec in frozen["spaces"].items():
        lines += [
            f"### {space}",
            "",
            f"- penalty: l1_ratio = {spec['l1_ratio']}, "
            f"alpha = {spec['alpha']:.6f}",
            f"- non-zero coefficients: {len(spec['coefficients'])}",
            f"- risk threshold (median of the derivation score): "
            f"{spec['risk_threshold']:.4f}",
            "",
        ]
        if spec["coefficients"]:
            lines.append("| Feature | Coefficient | HR per SD |")
            lines.append("|---|---|---|")
            for name, value in sorted(spec["coefficients"].items(),
                                      key=lambda kv: -abs(kv[1])):
                lines.append(f"| {name} | {value:+.4f} | {np.exp(value):.3f} |")
            lines.append("")

    path = cfg.RESULTS_DIR / "validation_protocol.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Protocol for the validating centre: {path}")


def _transform(X: np.ndarray, spec: dict) -> np.ndarray:
    """Reproduce the frozen preprocessing on new data."""
    X = np.asarray(X, dtype=float)

    X = X[:, spec["variance_kept"]]

    continuous = spec["scaler_continuous_indices"]
    if continuous and spec["scaler_mean"]:
        mean = np.asarray(spec["scaler_mean"])
        scale = np.asarray(spec["scaler_scale"])
        X[:, continuous] = (X[:, continuous] - mean) / scale

    return X[:, spec["correlation_kept"]]


def apply_to_external(clinical_path, radiomic_path, space="mixed_shape",
                      data_lock=None) -> dict:
    """Score an external cohort with the frozen model and report performance."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from sksurv.metrics import concordance_index_censored

    if not MODEL_PATH.exists():
        sys.exit(f"No frozen model at {MODEL_PATH}. Run `freeze` first.")
    frozen = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    spec = frozen["spaces"][space]

    original_clinical = cfg.DATA_CLINICAL
    original_radiomic = cfg.DATA_RADIOMIC
    original_lock = cfg.DATA_LOCK
    cfg.DATA_CLINICAL = Path(clinical_path)
    cfg.DATA_RADIOMIC = Path(radiomic_path)
    if data_lock:
        cfg.DATA_LOCK = data_lock

    try:
        from data_cox import load_dataset

        ds = load_dataset()
    finally:
        cfg.DATA_CLINICAL = original_clinical
        cfg.DATA_RADIOMIC = original_radiomic
        cfg.DATA_LOCK = original_lock

    columns = ds.columns_for(space)
    if columns != spec["feature_order"]:
        missing = set(spec["feature_order"]) - set(columns)
        extra = set(columns) - set(spec["feature_order"])
        if missing:
            sys.exit(f"External cohort is missing {len(missing)} features, "
                     f"for example {sorted(missing)[:5]}")
        print(f"  reordering columns to match the frozen model "
              f"({len(extra)} extra dropped)")
        X = ds.df[spec["feature_order"]].to_numpy(dtype=float)
    else:
        X = ds.X(space)

    X_transformed = _transform(X, spec)

    kept_names = [spec["feature_order"][i] for i in spec["variance_kept"]]
    kept_names = [kept_names[i] for i in spec["correlation_kept"]]
    coefficients = np.array([spec["coefficients"].get(n, 0.0)
                             for n in kept_names])

    score = X_transformed @ coefficients

    c_index = float(concordance_index_censored(
        ds.event.astype(bool), ds.duration, score)[0])

    high = score > spec["risk_threshold"]
    curves = {}
    for label, mask in (("High risk", high), ("Low risk", ~high)):
        if mask.sum() < 3:
            continue
        kmf = KaplanMeierFitter().fit(ds.duration[mask], ds.event[mask])
        curves[label] = {
            "n": int(mask.sum()),
            "events": int(ds.event[mask].sum()),
            "median": float(kmf.median_survival_time_),
        }

    logrank_p = np.nan
    if high.sum() >= 3 and (~high).sum() >= 3:
        logrank_p = float(logrank_test(
            ds.duration[high], ds.duration[~high],
            ds.event[high], ds.event[~high]).p_value)

    result = {
        "space": space,
        "n": int(len(ds.df)),
        "events": int(ds.n_events),
        "c_index_external": c_index,
        "c_index_apparent_derivation": spec["apparent_c_index"],
        "score_mean_external": float(score.mean()),
        "score_mean_derivation": spec["score_mean"],
        "risk_threshold": spec["risk_threshold"],
        "curves": curves,
        "logrank_p": logrank_p,
    }

    print(f"\n{'=' * 66}")
    print(f"External validation — {space}")
    print(f"{'=' * 66}")
    print(f"  Patients / events     : {result['n']} / {result['events']}")
    print(f"  C-index, external     : {c_index:.3f}")
    print(f"  C-index, derivation   : {spec['apparent_c_index']:.3f} (apparent)")
    print(f"  Score mean            : {score.mean():+.3f} external vs "
          f"{spec['score_mean']:+.3f} derivation")
    for label, info in curves.items():
        print(f"  {label:10s}: n = {info['n']}, {info['events']} events, "
              f"median {info['median']:.1f} months")
    if not np.isnan(logrank_p):
        print(f"  Log-rank p            : {logrank_p:.4f}")

    shift = abs(score.mean() - spec["score_mean"])
    if shift > 2 * max(spec["score_sd"], 1e-6):
        print("\n  NOTE: the external score distribution is shifted relative "
              "to the derivation cohort; recalibration of absolute risks will "
              "be needed even if discrimination holds.")

    out = cfg.RESULTS_DIR / f"external_validation_{space}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  saved: {out.name}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--spaces", nargs="+", default=None)

    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--clinical", required=True)
    apply_parser.add_argument("--radiomic", required=True)
    apply_parser.add_argument("--space", default="mixed_shape")
    apply_parser.add_argument("--data-lock", default=None,
                              help="date of last outcome ascertainment at the "
                                   "external centre, YYYY-MM-DD")

    args = parser.parse_args()

    if args.command == "freeze":
        freeze(spaces=args.spaces)
    else:
        apply_to_external(args.clinical, args.radiomic, args.space,
                          args.data_lock)
    return 0


if __name__ == "__main__":
    sys.exit(main())
