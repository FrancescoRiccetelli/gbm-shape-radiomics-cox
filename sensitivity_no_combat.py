"""Sensitivity analysis of the main results without scanner harmonisation."""
import config_cox as cfg

import json
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

TAG = "nocombat"
SPACES = ["clinical", "radiomic_shape", "mixed_shape"]


def main() -> int:
    import checks_cox as ck
    from data_cox import load_dataset
    from evaluation_cox import likelihood_ratio_rad_score, repeated_cv, summarise

    out_dir = cfg.RESULTS_DIR / "sensitivity_no_combat"
    out_dir.mkdir(parents=True, exist_ok=True)

    original = cfg.SCANNER_CORRECTION
    cfg.SCANNER_CORRECTION = "none"
    try:
        ds = load_dataset()
    finally:
        cfg.SCANNER_CORRECTION = original

    print(f"\nRepeated nested CV without ComBat ({cfg.N_REPEATS} x "
          f"{cfg.N_SPLITS_OUTER} folds, spaces: {', '.join(SPACES)})")
    folds = repeated_cv(ds, SPACES, tag=TAG)
    summary = summarise(folds)

    main_path = cfg.RESULTS_DIR / "cox_cv_summary.csv"
    if main_path.exists():
        main = pd.read_csv(main_path)[["space", "c_test_mean"]]
        main = main.rename(columns={"c_test_mean": "c_cv_combat"})
        summary = summary.merge(main, on="space", how="left")
        summary["difference"] = summary["c_test_mean"] - summary["c_cv_combat"]
    summary = summary.rename(columns={"c_test_mean": "c_cv_no_combat"})
    summary.to_csv(out_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")

    print("\n--- Cross-validated C-index ---")
    columns = [c for c in ("space", "c_cv_no_combat", "c_test_sd", "c_cv_combat",
                           "difference") if c in summary.columns]
    print(summary[columns].round(4).to_string(index=False))

    lr = likelihood_ratio_rad_score(ds, ck.oof_scores(ds, "radiomic_shape", TAG))
    lr = {k: v for k, v in lr.items() if k != "model_combined"}
    with open(out_dir / "likelihood_ratio_oof.json", "w", encoding="utf-8") as fh:
        json.dump(lr, fh, indent=2, default=str)
    print("\n--- Complementarity (out-of-fold) ---")
    print(f"  rad-score adds to clinical: chi2 = {lr['lr_score_added']:.2f}, "
          f"p = {lr['p_score_added']:.4g}")
    print(f"  clinical adds to rad-score: chi2 = {lr['lr_clinical_added']:.2f}, "
          f"p = {lr['p_clinical_added']:.4g}")
    print(f"  C-index combined (apparent): {lr['c_combined']:.3f}")

    windows = ck.landmark(ds, SPACES, tag=TAG)
    windows.to_csv(out_dir / "windows.csv", index=False, encoding="utf-8-sig")
    print("\n--- C-index by window ---")
    print(windows.round(3).to_string(index=False))

    crossover = ck.crossover_test(ds, SPACES, tag=TAG)
    crossover.to_csv(out_dir / "crossover.csv", index=False, encoding="utf-8-sig")
    print("\n--- Crossover ---")
    print(crossover[["horizon_months", "advantage_early", "advantage_survivors",
                     "crossover", "ci_lower", "ci_upper", "bootstrap_p"]]
          .to_string(index=False))

    print(f"\nsaved in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
