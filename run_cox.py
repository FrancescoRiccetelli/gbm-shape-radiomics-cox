"""Entry point that runs the analysis stages of the derivation cohort."""
from __future__ import annotations

import config_cox as cfg

import argparse
import json
import sys
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    path = cfg.RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved: {path.name}")


def save_json(obj: dict, name: str) -> None:
    with open(cfg.RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"  saved: {name}.json")


def stage_diagnostics(ds):
    """Cohort description and feature-space sizes."""
    from data_cox import cohort_summary

    banner("STAGE 1 - Cohort")
    table = cohort_summary(ds)
    print(table.to_string(index=False))
    save_table(table, "cox_cohort")
    return table


def stage_cv(ds, spaces, n_repeats):
    """Repeated nested cross-validation."""
    from evaluation_cox import (
        compare_all_spaces, compare_spaces, repeated_cv, selection_frequency,
        summarise,
    )

    banner(f"STAGE 2 - Repeated nested CV ({n_repeats} partitions)")
    folds = repeated_cv(ds, spaces=spaces, n_repeats=n_repeats)

    summary = summarise(folds)
    print("\n--- Mean C-index ---")
    print(summary.to_string(index=False))
    save_table(summary, "cox_cv_summary")

    frequency = selection_frequency(ds)
    if not frequency.empty:
        save_table(frequency, "cox_selection_frequency")
        top = frequency[frequency["space"] == "mixed_shape"].head(15)
        print("\n--- Most stable features (mixed) ---")
        print(top[["feature", "frequency", "direction"]].to_string(index=False))

    comparison = compare_spaces(ds, folds)
    if not comparison.empty:
        print("\n--- Nadeau-Bengio corrected t-test (primary spaces) ---")
        print(comparison.round(4).to_string(index=False))
        save_table(comparison, "cox_space_comparison")

    if folds[folds["status"] == "ok"]["space"].nunique() > len(cfg.FEATURE_SPACES):
        every = compare_all_spaces(ds, folds)
        save_table(every, "cox_space_comparison_all")

    return folds, summary, frequency


def stage_supplementary(ds, spaces):
    """Optimism correction, final model, radiomic score, stratification."""
    from evaluation_cox import (
        final_model, likelihood_ratio_rad_score, optimism_corrected,
        rad_score_table,
    )
    from analyses_cox import (
        calibration_at, orthogonality, shap_values, stratify_by_risk,
        time_dependent_auc,
    )

    banner("STAGE 3 - Final model and supplementary analyses")
    out = {}

    print("\n--- Optimism correction (bootstrap, a few minutes) ---")
    rows = []
    for space in spaces:
        try:
            apparent, optimism, corrected = optimism_corrected(ds, space)
            rows.append({"space": space, "Model": cfg.SPACE_LABELS.get(space),
                         "c_apparent": round(apparent, 4),
                         "optimism": round(optimism, 4),
                         "c_corrected": round(corrected, 4)})
            print(f"  {space:14s} apparent {apparent:.3f}  "
                  f"optimism {optimism:+.3f}  corrected {corrected:.3f}")
        except Exception:
            print(f"  {space}: failed")
            traceback.print_exc(limit=2)
    optimism_table = pd.DataFrame(rows)
    if not optimism_table.empty:
        path = cfg.RESULTS_DIR / "cox_optimism.csv"
        if path.exists():
            previous = pd.read_csv(path)
            optimism_table = (
                pd.concat([previous, optimism_table])
                .drop_duplicates("space", keep="last")
                .reset_index(drop=True))
        order = list(cfg.FEATURE_SPACES) + list(cfg.SECONDARY_SPACES)
        optimism_table["_o"] = optimism_table["space"].apply(
            lambda s: order.index(s) if s in order else len(order))
        optimism_table = (optimism_table.sort_values("_o")
                          .drop(columns="_o"))
        save_table(optimism_table, "cox_optimism")
    out["optimism"] = optimism_table

    print("\n--- Does the radiomic score already encode the clinical data? ---")
    try:
        orth = orthogonality(ds)
        if not orth.empty:
            save_table(orth, "cox_orthogonality")
            print(orth.to_string(index=False))
            out["orthogonality"] = orth
    except Exception:
        print("  orthogonality check failed:")
        traceback.print_exc(limit=2)

    print("\n--- Time-dependent AUC (averaged over folds) ---")
    try:
        tdauc = time_dependent_auc(ds)
        if not tdauc.empty:
            save_table(tdauc, "cox_time_dependent_auc")
            wide = tdauc.pivot(index="Model", columns="horizon_months",
                               values="auc_mean")
            print(wide.round(3).to_string())
            out["time_auc"] = tdauc
    except Exception:
        print("  time-dependent AUC failed:")
        traceback.print_exc(limit=2)

    print("\n--- Final mixed model ---")
    try:
        model = final_model(ds, "mixed_shape")
        print(f"  features retained: {len(model['coefficients'])}")
        print(f"  apparent C-index : {model['c_index']:.3f}")

        table = rad_score_table(model)
        save_table(table, "cox_rad_score")
        print(table.head(15).to_string(index=False))
        out["model"] = model
        out["rad_score_table"] = table

        strat = stratify_by_risk(ds, model["score"])
        print(f"\n  risk stratification: log-rank p = {strat['logrank_p']:.2e}")
        for label, info in strat["curves"].items():
            print(f"    {label}: n = {info['n']}, "
                  f"median {info['median']:.1f} months")
        out["stratification"] = strat

        radiomic_space = next(
            (sp for sp in spaces if sp.startswith("radiomic")), None)
        radiomic_model = (final_model(ds, radiomic_space)
                          if radiomic_space else model)
        out["radiomic_model"] = radiomic_model
        lr = likelihood_ratio_rad_score(ds, radiomic_model["score"])
        print("\n--- Does each block add to the other? ---")
        print(f"  C-index clinical only  : {lr['c_clinical']:.3f}")
        print(f"  C-index rad-score only : {lr['c_rad_score']:.3f}")
        print(f"  C-index combined       : {lr['c_combined']:.3f}")
        print(f"  rad-score adds to clinical: chi2 = {lr['lr_score_added']:.2f}, "
              f"p = {lr['p_score_added']:.4f}")
        print(f"  clinical adds to rad-score: chi2 = {lr['lr_clinical_added']:.2f}, "
              f"p = {lr['p_clinical_added']:.4f}")
        save_json({"score_source": "in-sample final model (optimistic)",
                   **{k: v for k, v in lr.items() if k != "model_combined"}},
                  "cox_likelihood_ratio")
        out["lr"] = lr

        if radiomic_space:
            try:
                from checks_cox import oof_scores
                lr_oof = likelihood_ratio_rad_score(
                    ds, oof_scores(ds, radiomic_space))
                print("\n--- Same test, out-of-fold radiomic score (report this) ---")
                print(f"  C-index rad-score only : {lr_oof['c_rad_score']:.3f}")
                print(f"  C-index combined       : {lr_oof['c_combined']:.3f}")
                print(f"  rad-score adds to clinical: chi2 = "
                      f"{lr_oof['lr_score_added']:.2f}, "
                      f"p = {lr_oof['p_score_added']:.4f}")
                print(f"  clinical adds to rad-score: chi2 = "
                      f"{lr_oof['lr_clinical_added']:.2f}, "
                      f"p = {lr_oof['p_clinical_added']:.4f}")
                save_json({"score_source": f"out-of-fold ({radiomic_space}, "
                                           f"mean over CV repeats)",
                           **{k: v for k, v in lr_oof.items()
                              if k != "model_combined"}},
                          "cox_likelihood_ratio_oof")
                out["lr_oof"] = lr_oof
            except Exception:
                print("  out-of-fold likelihood ratio test failed "
                      "(run the cv stage first):")
                traceback.print_exc(limit=2)

        calibration = {}
        for horizon in cfg.TIME_HORIZONS:
            try:
                calibration[horizon] = calibration_at(ds, model, horizon)
            except Exception:
                pass
        if calibration:
            for horizon, table in calibration.items():
                save_table(table, f"cox_calibration_{int(horizon)}m")
            out["calibration"] = calibration

        if cfg.RUN_SHAP:
            try:
                out["shap"] = shap_values(ds, "mixed_shape", model)
            except Exception:
                print("  SHAP failed:")
                traceback.print_exc(limit=2)

    except Exception:
        print("  final model failed:")
        traceback.print_exc(limit=3)

    return out


def stage_checks(ds):
    """Assumptions, sensitivity analyses and clinical utility."""
    import checks_cox

    banner("STAGE 3b - Robustness checks")
    out = checks_cox.run_all(ds)

    for key, name in [("proportional_hazards", "cox_proportional_hazards"),
                      ("time_varying", "cox_time_varying_effect"),
                      ("drift", "cox_temporal_drift"),
                      ("subcohorts", "cox_field_subcohorts"),
                      ("biopsy", "cox_biopsy_sensitivity"),
                      ("reduced", "cox_reduced_model"),
                      ("dca", "cox_decision_curve"),
                      ("landmark", "cox_landmark"),
                      ("pair_types", "cox_pair_types"),
                      ("sample_size", "cox_sample_size"),
                      ("crossover", "cox_crossover"),
                      ("crossover_power", "cox_crossover_power"),
                      ("crossover_windows", "cox_crossover_windows"),
                      ("crossover_power", "cox_crossover_power"),
                      ("crossover_windows", "cox_crossover_windows"),
                      ("time_interaction", "cox_time_interaction")]:
        table = out.get(key)
        if isinstance(table, pd.DataFrame) and not table.empty:
            save_table(table, name)
    if out.get("stratified"):
        save_table(out["stratified"]["summary"], "cox_stratified_model")
        save_json({k: v for k, v in out["stratified"].items()
                   if k != "summary"}, "cox_stratified_cindex")
    return out


def stage_montecarlo(ds):
    """Paired random splits."""
    import checks_cox

    banner("STAGE 3c - Monte Carlo cross-validation")
    spaces = list(cfg.FEATURE_SPACES)
    results = checks_cox.monte_carlo(ds, spaces)
    summary = checks_cox.monte_carlo_summary(results, spaces)
    print(summary.to_string(index=False))
    save_table(results, "cox_montecarlo_splits")
    save_table(summary, "cox_montecarlo")
    return {"results": results, "summary": summary}


def stage_permutation(ds, spaces):
    """Permutation testing."""
    from analyses_cox import permutation_test

    banner(f"STAGE 4 - Permutation testing ({cfg.N_PERMUTATIONS} permutations)")
    results = {}
    for space in spaces:
        print(f"\n--- {space} ---")
        try:
            results[space] = permutation_test(ds, space)
        except Exception:
            print(f"  {space}: failed")
            traceback.print_exc(limit=2)

    if results:
        save_table(pd.DataFrame([
            {"Model": cfg.SPACE_LABELS.get(s), "C-index": round(r["observed"], 4),
             "p-value": round(r["p_value"], 4), "Permutations": r["n"],
             "Finest resolvable p": round(r["resolution"], 4),
             "Failed": r["n_failed"]}
            for s, r in results.items()]), "cox_permutation")
    return results


def stage_figures(ds, folds, summary, frequency, supplementary, permutation,
                  checks=None):
    """Draw all figures."""
    import figures_cox as fg

    banner("STAGE 5 - Figures")

    if summary is not None and not summary.empty:
        optimism = supplementary.get("optimism")
        fg.plot_cindex(summary, optimism, name="fig_cox_cindex_all_spaces")
        fg.plot_cindex(summary, optimism, spaces=list(cfg.FEATURE_SPACES),
                       name="fig_cox_cindex", show_title=False)
    if folds is not None:
        fg.plot_cindex_distribution(folds)
    if frequency is not None and not frequency.empty:
        fg.plot_selection_frequency(frequency)
    if supplementary.get("stratification"):
        fg.plot_risk_stratification(supplementary["stratification"], ds)
    if supplementary.get("rad_score_table") is not None:
        fg.plot_rad_score_forest(supplementary["rad_score_table"])
    if supplementary.get("calibration"):
        fg.plot_calibration(supplementary["calibration"])
    if permutation:
        fg.plot_permutation(permutation)
    if supplementary.get("orthogonality") is not None:
        fg.plot_orthogonality(supplementary["orthogonality"])
    if supplementary.get("time_auc") is not None:
        fg.plot_time_dependent_auc(supplementary["time_auc"])
    if supplementary.get("shap"):
        fg.plot_shap(supplementary["shap"])
    checks = checks or {}
    if checks.get("dca") is not None:
        fg.plot_decision_curve(checks["dca"])
    if checks.get("proportional_hazards") is not None:
        fg.plot_proportional_hazards(checks["proportional_hazards"])
    if checks.get("time_varying") is not None:
        fg.plot_time_varying(checks["time_varying"])
    if checks.get("landmark") is not None:
        fg.plot_landmark(checks["landmark"])
    if checks.get("crossover") is not None:
        fg.plot_crossover(checks["crossover"])
        if checks.get("crossover_windows") is not None:
            fg.plot_crossover_lines(checks["crossover_windows"],
                                    checks["crossover"])
    if checks.get("pair_types") is not None:
        fg.plot_pair_types(checks["pair_types"])
    if checks.get("sample_size") is not None:
        fg.plot_sample_size(checks["sample_size"])
    if checks.get("montecarlo") is not None:
        fg.plot_montecarlo(checks["montecarlo"]["results"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multiregional penalised-Cox analysis")
    parser.add_argument("--stage", default="all",
                        choices=["all", "diagnostics", "cv", "supplementary",
                                 "checks", "montecarlo", "permutation",
                                 "figures"])
    parser.add_argument("--dry-run", action="store_true",
                        help="time a few permutations and estimate the cost")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--spaces", nargs="+", default=None,
                        help="feature spaces to run (default: the three primary)")
    parser.add_argument("--secondary", action="store_true",
                        help="also run the single-region spaces")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cfg.clear_cache()
    if args.repeats is not None:
        cfg.N_REPEATS = args.repeats
    if args.permutations is not None:
        cfg.N_PERMUTATIONS = args.permutations

    spaces = args.spaces or list(cfg.FEATURE_SPACES)
    if args.secondary:
        spaces += list(cfg.SECONDARY_SPACES)

    started = time.time()
    banner(f"PENALISED COX PIPELINE - {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(cfg.describe())
    if cfg.USE_CACHE:
        print(f"Cache on disk      : {cfg.cache_size_mb():.0f} MB")
    print(f"Spaces             : {', '.join(spaces)}")

    from data_cox import load_dataset

    ds = load_dataset()

    if args.dry_run:
        from analyses_cox import dry_run

        banner("DRY RUN")
        dry_run(ds)
        return 0

    folds = summary = frequency = None
    supplementary, permutation, checks = {}, {}, {}

    try:
        if args.stage in ("all", "diagnostics"):
            stage_diagnostics(ds)

        if args.stage in ("all", "cv", "supplementary", "figures"):
            folds, summary, frequency = stage_cv(ds, spaces, cfg.N_REPEATS)

        if args.stage in ("all", "supplementary", "figures"):
            supplementary = stage_supplementary(ds, spaces)

        if args.stage in ("all", "checks", "figures"):
            checks = stage_checks(ds)

        if args.stage in ("all", "montecarlo", "figures"):
            checks["montecarlo"] = stage_montecarlo(ds)

        if args.stage in ("all", "permutation", "figures"):
            permutation = stage_permutation(
                ds, [s for s in spaces if s in cfg.FEATURE_SPACES])

        if args.stage in ("all", "figures"):
            stage_figures(ds, folds, summary, frequency, supplementary,
                          permutation, checks)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Completed work is on disk; rerun to resume.")
        return 130

    banner(f"COMPLETE - {(time.time() - started) / 60:.1f} minutes")
    print(f"Tables      : {cfg.RESULTS_DIR}")
    print(f"Figures     : {cfg.FIGURES_DIR}")
    print(f"Checkpoints : {cfg.CKPT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
