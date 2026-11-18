"""External validation of the frozen models on an independent cohort."""
from __future__ import annotations

import config_cox as cfg

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from freeze_model import MODEL_PATH, _transform

OUT_DIR = cfg.RESULTS_DIR / "external_validation"


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip()
                         .str.replace("%", "", regex=False)
                         .str.replace(",", ".", regex=False), errors="coerce")


def load_external(clinical_path, radiomic_path, data_lock, frozen) -> pd.DataFrame:
    """Read the external cohort and recode it to the derivation conventions."""
    from data_cox import add_volume_ratios

    raw = (pd.read_excel(clinical_path) if str(clinical_path).endswith(".xlsx")
           else pd.read_csv(clinical_path))

    df = pd.DataFrame()
    id_column = "ID_Studio" if "ID_Studio" in raw.columns else cfg.ID_COL
    df[cfg.ID_COL] = pd.to_numeric(
        raw[id_column].astype(str).str.extract(r"^(\d+)")[0])

    df["Eta"] = _number(raw["Età"])
    df["GENERE"] = raw["GENERE"].astype(str).str.strip().str.upper().map(
        {"M": 0, "F": 1})
    df["RESEZIONE"] = (raw["RESEZIONE"].astype(str).str.strip().str.upper()
                       == "GTR").astype(int)

    ki67 = _number(raw["Ki67%"] if "Ki67%" in raw.columns else raw["Ki67"])
    df["Ki67"] = np.where(ki67 <= 1, ki67 * 100, ki67)

    df["MGMT_METILAZIONE"] = _number(raw["MGMT_METILAZIONE"])

    p53 = raw["p53_ESPRESSIONE"].astype(str).str.strip().str.lower()
    aberrant = set(frozen.get("p53_aberrant_categories", cfg.P53_ABERRANT_CATEGORIES))
    df["p53_ESPRESSIONE"] = (~p53.isin(aberrant)).astype(float)
    df.loc[p53.isin(("nan", "", "na")), "p53_ESPRESSIONE"] = np.nan

    location_column = next((c for c in raw.columns
                            if str(c).strip().lower() in ("sede", "lobo", "location")),
                           None)
    if location_column:
        location = raw[location_column].fillna("").astype(str).str.upper()
        df["DEEP"] = location.str.contains(frozen["deep_pattern"],
                                           regex=True).astype(float)
    else:
        df["DEEP"] = np.nan

    alive = raw["Vivo"].astype(str).str.strip().str.lower().isin(("yes", "si", "sì"))
    df[cfg.EVENT_COL] = (~alive).astype(int)
    surgery = pd.to_datetime(raw["Date of surgery"], format="%d/%m/%Y",
                             errors="coerce")
    to_lock = (pd.Timestamp(data_lock) - surgery).dt.days / 30.4375
    df[cfg.DURATION_COL] = np.where(df[cfg.EVENT_COL] == 1,
                                    _number(raw["OS"]), to_lock)

    radiomic = pd.read_csv(radiomic_path)
    if cfg.ADD_VOLUME_RATIOS:
        radiomic = add_volume_ratios(radiomic)

    merged = df.merge(radiomic, on=cfg.ID_COL, how="inner")
    merged = merged[merged[cfg.DURATION_COL] > 0].reset_index(drop=True)

    print(f"  clinical rows: {len(df)} | with radiomics: {len(merged)} | "
          f"deaths: {int(merged[cfg.EVENT_COL].sum())}")
    return merged


def harmonise(frame: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Map the external feature distribution onto the derivation reference."""
    spec = frozen.get("combat")
    if frozen.get("scanner_correction") != "combat" or not spec:
        return frame

    out = frame.copy()
    columns = [c for c in spec["columns"] if c in out.columns]
    mean = pd.Series(spec["pooled_mean"], index=spec["columns"])[columns]
    sd = pd.Series(spec["pooled_sd"], index=spec["columns"])[columns]

    own_mean = out[columns].mean()
    own_sd = out[columns].std(ddof=0).replace(0, 1.0)
    out[columns] = (out[columns] - own_mean) / own_sd * sd + mean
    print(f"  harmonised {len(columns)} features to the derivation reference")
    return out


def score_space(frame: pd.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen transformation and coefficients; return score and mask."""
    X = frame[spec["feature_order"]].to_numpy(dtype=float)
    usable = ~np.isnan(X).any(axis=1)

    Xt = _transform(X[usable], spec)
    names = [spec["feature_order"][i] for i in spec["variance_kept"]]
    names = [names[i] for i in spec["correlation_kept"]]
    beta = np.array([spec["coefficients"].get(n, 0.0) for n in names])
    return Xt @ beta, usable


def model_based_concordance(lp: np.ndarray) -> float:
    """Concordance expected from the linear predictor alone (van Klaveren et al.)."""
    lp = np.asarray(lp, dtype=float)
    diff = np.abs(lp[:, None] - lp[None, :])
    upper = diff[np.triu_indices(len(lp), k=1)]
    return float(np.mean(1.0 / (1.0 + np.exp(-upper))))


def pair_decomposition(duration, event, score, horizon=12.0) -> dict:
    """Harrell's C restricted to pairs of a given type."""
    early = (duration < horizon) & (event == 1)
    out = {}
    for label, first, second in (("early_vs_late", early, ~early),
                                 ("late_vs_late", ~early, ~early),
                                 ("early_vs_early", early, early)):
        concordant = total = 0.0
        for i in np.flatnonzero(first & (event == 1)):
            for j in np.flatnonzero(second):
                if duration[j] > duration[i]:
                    total += 1
                    concordant += ((score[i] > score[j])
                                   + 0.5 * (score[i] == score[j]))
        out[label] = {"c_index": float(concordant / total) if total else None,
                      "pairs": int(total)}
    return out


def evaluate(duration, event, score, threshold, n_boot=2000, seed=42) -> dict:
    """Discrimination, calibration slope and risk stratification."""
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
    from sksurv.util import Surv

    def cidx(d, e, s):
        return float(concordance_index_censored(e.astype(bool), d, s)[0])

    c = cidx(duration, event, score)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(score), len(score))
        if event[i].sum() < 5:
            continue
        boot.append(cidx(duration[i], event[i], score[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    cal = CoxPHFitter().fit(pd.DataFrame({"score": score, "duration": duration,
                                          "event": event}),
                            duration_col="duration", event_col="event")
    slope = cal.summary.loc["score"]

    sd = score.std()
    hr_frame = pd.DataFrame({"score_sd": score / sd if sd > 0 else score,
                             "duration": duration, "event": event})
    hr = CoxPHFitter().fit(hr_frame, duration_col="duration",
                           event_col="event").summary.loc["score_sd"]

    high = score > threshold
    groups = {}
    for label, mask in (("High risk", high), ("Low risk", ~high)):
        if mask.sum() >= 3:
            km = KaplanMeierFitter().fit(duration[mask], event[mask])
            groups[label] = {"n": int(mask.sum()), "events": int(event[mask].sum()),
                             "median": float(km.median_survival_time_)}
    logrank = (float(logrank_test(duration[high], duration[~high],
                                  event[high], event[~high]).p_value)
               if high.sum() >= 3 and (~high).sum() >= 3 else np.nan)

    y = Surv.from_arrays(event=event.astype(bool), time=duration)
    auc = {}
    for h in (6.0, 12.0):
        if (duration < h).sum() >= 3 and (duration >= h).sum() >= 3:
            try:
                auc[f"{h:.0f}m"] = float(cumulative_dynamic_auc(y, y, score, [h])[0][0])
            except Exception:
                pass

    return {
        "n": int(len(score)), "events": int(event.sum()),
        "c_index": c, "c_index_ci": [float(lo), float(hi)],
        "c_expected": model_based_concordance(score),
        "pairs": pair_decomposition(duration, event, score),
        "hr_per_sd": float(hr["exp(coef)"]),
        "hr_per_sd_ci": [float(hr["exp(coef) lower 95%"]),
                         float(hr["exp(coef) upper 95%"])],
        "hr_per_sd_p": float(hr["p"]),
        "calibration_slope": float(slope["coef"]),
        "calibration_slope_ci": [float(slope["coef lower 95%"]),
                                 float(slope["coef upper 95%"])],
        "risk_groups": groups, "logrank_p": logrank,
        "time_dependent_auc": auc,
        "high_risk_mask": high.tolist(),
    }


def km_figure(duration, event, high, result, space, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, mask, colour in (("High risk", high, "#C44E52"),
                                ("Low risk", ~high, "#4C72B0")):
        if mask.sum() < 3:
            continue
        KaplanMeierFitter().fit(duration[mask], event[mask], label=label) \
            .plot_survival_function(ax=ax, color=colour, linewidth=2,
                                    ci_show=True)
    ax.axvline(12, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Time since diagnosis (months)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.02)
    p = result["logrank_p"]
    p_text = ("not computable" if p is None or np.isnan(p)
              else "p < 0.001" if p < 0.001 else f"p = {p:.3f}")
    ax.set_title(f"External validation \u2013 {cfg.SPACE_LABELS.get(space, space)} model\n"
                 f"Frozen threshold; log-rank {p_text}; "
                 f"C-index {result['c_index']:.3f}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def follow_up_windows(duration, event, score, horizons=(6.0, 12.0),
                      n_boot=1000, seed=42) -> list[dict]:
    """C-index in an early and a late window of follow-up, with bootstrap CIs."""
    from sksurv.metrics import concordance_index_censored

    duration = np.asarray(duration, float)
    event = np.asarray(event, int)
    score = np.asarray(score, float)

    def cidx(d, e, s):
        if e.sum() < 1:
            return np.nan
        return float(concordance_index_censored(e.astype(bool), d, s)[0])

    def windows(d, e, s):
        out = []
        for k, t in enumerate(sorted(horizons)):
            ev = (d < t) & (e == 1)
            out.append((k, f"First {t:.0f} months", int(ev.sum()),
                        cidx(np.minimum(d, t), ev, s)))
        out.append((len(horizons), "Whole follow-up", int(e.sum()),
                    cidx(d, e, s)))
        for k, t in enumerate(sorted(horizons)):
            alive = d >= t
            out.append((len(horizons) + 1 + k, f"Alive at {t:.0f} months",
                        int(e[alive].sum()),
                        cidx(d[alive] - t, e[alive], s[alive])
                        if alive.sum() >= 5 else np.nan))
        return out

    observed = windows(duration, event, score)
    rng = np.random.default_rng(seed)
    boot = [[] for _ in observed]
    for _ in range(n_boot):
        i = rng.integers(0, len(score), len(score))
        for j, (_, _, _, value) in enumerate(windows(duration[i], event[i],
                                                      score[i])):
            if not np.isnan(value):
                boot[j].append(value)

    rows = []
    for (order, label, events, value), draws in zip(observed, boot):
        lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) > 50
                  else (np.nan, np.nan))
        rows.append({"order": order, "population": label, "events": events,
                     "c_index": value, "ci_lower": float(lo),
                     "ci_upper": float(hi)})
    return rows


def windows_figure(derivation: pd.DataFrame, external: pd.DataFrame,
                   min_events: int = 10) -> None:
    """Discrimination by window of follow-up, derivation and external side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"clinical": "#DD8452", "radiomic_shape": "#4C72B0",
               "mixed_shape": "#55A868"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, (title, table) in zip(axes, (("Derivation cohort (out-of-fold)",
                                          derivation),
                                         ("External cohort (frozen model)",
                                          external))):
        order = (table.drop_duplicates("population").sort_values("order")
                 ["population"].tolist())
        x = np.arange(len(order))
        spaces = [s for s in colours if s in set(table["space"])]
        for k, space in enumerate(spaces):
            block = table[table["space"] == space].set_index("population")
            block = block.reindex(order)
            dx = (k - (len(spaces) - 1) / 2) * 0.12
            colour = colours[space]
            ax.plot(x + dx, block["c_index"], color=colour, linewidth=2,
                    label=cfg.SPACE_LABELS.get(space, space))
            for xi, (_, r) in zip(x + dx, block.iterrows()):
                solid = r["events"] >= min_events
                if not np.isnan(r["ci_lower"]):
                    ax.plot([xi, xi], [r["ci_lower"], r["ci_upper"]],
                            color=colour, linewidth=1.2, alpha=0.6)
                ax.plot(xi, r["c_index"], "o", markersize=8, color=colour,
                        markerfacecolor=colour if solid else "white",
                        markeredgewidth=1.8)
        early = sum(p.startswith("First") for p in order)
        ax.axvspan(-0.5, early - 0.5, color="#C44E52", alpha=0.06)
        ax.axvspan(early + 0.5, len(order) - 0.5, color="#4C72B0", alpha=0.06)
        counts = table.drop_duplicates(["population", "space"])
        labels = []
        for p in order:
            ev = counts[counts.population == p].set_index("space")["events"]
            labels.append(p + "\n" + " / ".join(str(int(ev[s]))
                                                 for s in spaces if s in ev))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Harrell's C-index")
    axes[0].set_ylim(0.2, 1.0)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Discrimination by window of follow-up\n"
                 "Deaths per window under each label (clinical / radiomic / "
                 f"mixed); hollow points rest on fewer than {min_events} "
                 "deaths; bars are bootstrap 95% CIs", y=1.03, fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / "fig_validation_windows.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def pair_figure(external: pd.DataFrame, spaces,
                derivation: pd.DataFrame | None = None) -> None:
    """Pair-type concordance side by side, derivation against external."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if derivation is None:
        from checks_cox import pair_types
        from data_cox import load_dataset
        derivation = pair_types(load_dataset(), spaces)
    external = external[external["subset"] == "common"]
    kinds = ["early_vs_early", "early_vs_late", "late_vs_late"]
    names = {"early_vs_early": "Early vs early", "early_vs_late": "Early vs late",
             "late_vs_late": "Late vs late"}
    shown = [s for s in spaces if s in set(external["space"])]

    fig, axes = plt.subplots(1, len(shown), figsize=(4.6 * len(shown), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, space in zip(axes, shown):
        d = derivation[derivation.space == space].set_index("pair_type")
        e = external[external.space == space].set_index("pair_type")
        x = np.arange(len(kinds))
        ax.bar(x - 0.2, [d.loc[k, "c_index"] for k in kinds], 0.4,
               color="#4C72B0", label="Derivation (out-of-fold)")
        ax.bar(x + 0.2, [e.loc[k, "c_index"] if k in e.index else np.nan
                         for k in kinds], 0.4, color="#DD8452",
               label="External")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{names[k]}\n({int(d.loc[k, 'pairs'])} / "
                            f"{int(e.loc[k, 'pairs']) if k in e.index else 0})"
                            for k in kinds], fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(cfg.SPACE_LABELS.get(space, space))
        ax.set_ylim(0.3, 0.95)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("C-index within pair type")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Concordance by pair type, derivation versus external cohort\n"
                 "Numbers of pairs: derivation / external", y=1.04)
    fig.tight_layout()
    path = OUT_DIR / "fig_validation_pair_types.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--radiomic", required=True)
    parser.add_argument("--data-lock", default=cfg.DATA_LOCK)
    parser.add_argument("--spaces", nargs="+", default=None)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"No frozen model at {MODEL_PATH}. Run freeze_model.py freeze.")
    frozen = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading external cohort...")
    frame = load_external(args.clinical, args.radiomic, args.data_lock, frozen)
    frame = harmonise(frame, frozen)

    spaces = args.spaces or list(frozen["spaces"])

    usable = {}
    for space in spaces:
        spec = frozen["spaces"][space]
        cols = [c for c in spec["feature_order"] if c in frame.columns]
        usable[space] = (frame[cols].notna().all(axis=1).to_numpy()
                         if len(cols) == len(spec["feature_order"])
                         else np.zeros(len(frame), bool))
    common = np.logical_and.reduce([m for m in usable.values() if m.any()])

    rows, pair_rows = [], []
    for subset, label in (("own", "all patients scorable by the model"),
                          ("common", "patients scorable by every model")):
        print(f"\n##### {label} #####")
        for space in spaces:
            spec = frozen["spaces"][space]
            mask = usable[space] if subset == "own" else common & usable[space]
            print(f"\n=== {cfg.SPACE_LABELS.get(space, space)} ===")
            if not mask.any():
                print("  not applicable: required variables unavailable")
                rows.append({"space": space, "subset": subset,
                             "status": "not applicable"})
                continue

            score, ok = score_space(frame[mask].reset_index(drop=True), spec)
            duration = frame[cfg.DURATION_COL].to_numpy(float)[mask][ok]
            event = frame[cfg.EVENT_COL].to_numpy(int)[mask][ok]
            if len(score) < 20 or event.sum() < 10:
                print(f"  too few: {len(score)} patients, {event.sum()} events")
                rows.append({"space": space, "subset": subset,
                             "status": "too few complete cases"})
                continue

            result = evaluate(duration, event, score, spec["risk_threshold"])
            high = np.array(result.pop("high_risk_mask"))
            ref = spec.get("derivation_expected_c")

            print(f"  patients {result['n']}, deaths {result['events']}")
            print(f"  C-index        {result['c_index']:.3f} (95% CI "
                  f"{result['c_index_ci'][0]:.3f}-{result['c_index_ci'][1]:.3f})")
            print(f"  C expected     {result['c_expected']:.3f}"
                  + (f"  (derivation: {ref:.3f})" if ref else ""))
            for kind, val in result["pairs"].items():
                if val["c_index"] is not None:
                    print(f"    {kind:15s} C = {val['c_index']:.3f}  "
                          f"({val['pairs']} pairs)")
                pair_rows.append({"space": space, "subset": subset,
                                  "pair_type": kind, **val})
            print(f"  HR per SD      {result['hr_per_sd']:.2f} "
                  f"(p = {result['hr_per_sd_p']:.4f})")
            print(f"  calib. slope   {result['calibration_slope']:.2f}")
            for g, info in result["risk_groups"].items():
                print(f"  {g:10s}     n = {info['n']}, {info['events']} deaths, "
                      f"median {info['median']:.1f} months")
            print(f"  log-rank p     {result['logrank_p']:.3g}")
            if result["time_dependent_auc"]:
                print("  AUC            " + ", ".join(
                    f"{k} {x:.3f}" for k, x in result["time_dependent_auc"].items()))

            (OUT_DIR / f"validation_{space}_{subset}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
            km_figure(duration, event, high, result, space,
                      OUT_DIR / f"fig_validation_{space}_{subset}.png")
            rows.append({"space": space, "subset": subset, "status": "ok", **{
                k: x for k, x in result.items()
                if k not in ("risk_groups", "time_dependent_auc", "pairs")}})

    try:
        from checks_cox import oof_scores
        from data_cox import load_dataset

        ext_rows, der_rows = [], []
        for space in spaces:
            mask = common & usable[space]
            if not mask.any():
                continue
            spec = frozen["spaces"][space]
            sub = frame[mask].reset_index(drop=True)
            score, ok = score_space(sub, spec)
            for r in follow_up_windows(sub[cfg.DURATION_COL].to_numpy()[ok],
                                       sub[cfg.EVENT_COL].to_numpy()[ok], score):
                ext_rows.append({"space": space, **r})
        ds = load_dataset()
        for space in spaces:
            for r in follow_up_windows(ds.duration, ds.event,
                                       oof_scores(ds, space)):
                der_rows.append({"space": space, **r})
        ext, der = pd.DataFrame(ext_rows), pd.DataFrame(der_rows)
        ext.to_csv(OUT_DIR / "validation_windows_external.csv", index=False,
                   encoding="utf-8-sig")
        der.to_csv(OUT_DIR / "validation_windows_derivation.csv", index=False,
                   encoding="utf-8-sig")
        print("\n--- Discrimination by window of follow-up (external) ---")
        print(ext.pivot(index="space", columns="population", values="c_index")
              [ext.drop_duplicates("population").sort_values("order")
               ["population"]].round(3).to_string())
        windows_figure(der, ext)
    except Exception as exc:
        print(f"  window analysis skipped ({type(exc).__name__}: {exc})")

    pd.DataFrame(pair_rows).to_csv(OUT_DIR / "validation_pair_types.csv",
                                   index=False, encoding="utf-8-sig")
    try:
        pair_figure(pd.DataFrame(pair_rows), spaces)
    except Exception as exc:
        print(f"  pair figure skipped ({type(exc).__name__}: {exc})")

    pd.DataFrame(rows).to_csv(OUT_DIR / "validation_summary.csv", index=False,
                              encoding="utf-8-sig")
    print(f"\nResults in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
