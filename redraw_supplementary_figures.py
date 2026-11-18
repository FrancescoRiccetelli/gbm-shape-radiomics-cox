"""Redraws the supplementary figures from the saved result tables."""
import config_cox as cfg

import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

R = cfg.RESULTS_DIR
EXT = R / "external_validation"


def read(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def attempt(name, needed, draw):
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        print(f"  skipped {name}: missing {', '.join(missing)}")
        return
    try:
        draw()
    except Exception as exc:
        print(f"  FAILED {name}: {exc}")


def main() -> int:
    import figures_cox as fg

    spaces = list(cfg.FEATURE_SPACES)

    attempt("C-index, all spaces", [R / "cox_cv_summary.csv", R / "cox_optimism.csv"],
            lambda: fg.plot_cindex(read(R / "cox_cv_summary.csv"),
                                   read(R / "cox_optimism.csv"),
                                   name="fig_cox_cindex_all_spaces"))

    folds_path = cfg.CKPT_DIR / "cv_folds_main.csv"
    attempt("C-index distribution", [folds_path],
            lambda: fg.plot_cindex_distribution(read(folds_path)))

    attempt("selection frequency", [R / "cox_selection_frequency.csv"],
            lambda: fg.plot_selection_frequency(read(R / "cox_selection_frequency.csv")))

    attempt("coefficients", [R / "cox_rad_score.csv"],
            lambda: fg.plot_rad_score_forest(read(R / "cox_rad_score.csv")))

    def permutation():
        import numpy as np
        summary = read(R / "cox_permutation.csv")
        results = {}
        for space in spaces:
            path = cfg.CKPT_DIR / f"permutation_{space}.csv"
            observed = cfg.CKPT_DIR / f"permutation_observed_{space}.txt"
            if not (path.exists() and observed.exists()):
                continue
            table = read(path)
            null = table.loc[table["status"] == "ok", "c_index"].dropna().to_numpy()
            obs = float(observed.read_text().strip())
            results[space] = {"observed": obs, "null": null, "n": len(null),
                              "p_value": (1 + (null >= obs).sum()) / (1 + len(null)),
                              "resolution": 1 / (1 + len(null))}
        if not results:
            raise FileNotFoundError("no permutation checkpoints")
        for space, r in results.items():
            row = summary[summary["Model"].str.startswith(fg._label(space).split()[0])]
            if len(row) and abs(row["p-value"].iloc[0] - r["p_value"]) > 1e-3:
                print(f"  warning: permutation p for {space} differs from "
                      f"cox_permutation.csv")
        fg.plot_permutation(results)

    attempt("permutation", [R / "cox_permutation.csv"], permutation)

    attempt("Monte Carlo", [R / "cox_montecarlo_splits.csv"],
            lambda: fg.plot_montecarlo(read(R / "cox_montecarlo_splits.csv")))

    attempt("time-dependent AUC", [R / "cox_time_dependent_auc.csv"],
            lambda: fg.plot_time_dependent_auc(read(R / "cox_time_dependent_auc.csv")))

    attempt("proportional hazards", [R / "cox_proportional_hazards.csv"],
            lambda: fg.plot_proportional_hazards(read(R / "cox_proportional_hazards.csv")))

    attempt("crossover", [R / "cox_crossover_windows.csv", R / "cox_crossover.csv"],
            lambda: (fg.plot_crossover_lines(read(R / "cox_crossover_windows.csv"),
                                             read(R / "cox_crossover.csv")),
                     fg.plot_crossover(read(R / "cox_crossover.csv"))))

    attempt("orthogonality", [R / "cox_orthogonality.csv"],
            lambda: fg.plot_orthogonality(read(R / "cox_orthogonality.csv")))

    calibration = {h: R / f"cox_calibration_{int(h)}m.csv" for h in cfg.TIME_HORIZONS}
    attempt("calibration", list(calibration.values()),
            lambda: fg.plot_calibration({h: read(p) for h, p in calibration.items()}))

    attempt("decision curve", [R / "cox_decision_curve.csv"],
            lambda: fg.plot_decision_curve(read(R / "cox_decision_curve.csv")))

    attempt("sample size", [R / "cox_sample_size.csv"],
            lambda: fg.plot_sample_size(read(R / "cox_sample_size.csv")))

    def external():
        import validate_external as ve
        ve.windows_figure(read(EXT / "validation_windows_derivation.csv"),
                          read(EXT / "validation_windows_external.csv"))
        derivation = read(R / "cox_pair_types.csv")
        ve.pair_figure(read(EXT / "validation_pair_types.csv"), spaces,
                       derivation=derivation)

    attempt("external windows and pairs",
            [EXT / "validation_windows_derivation.csv",
             EXT / "validation_windows_external.csv",
             EXT / "validation_pair_types.csv", R / "cox_pair_types.csv"],
            external)

    print("\nExternal Kaplan-Meier curves: rerun step 10 (validate_external.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
