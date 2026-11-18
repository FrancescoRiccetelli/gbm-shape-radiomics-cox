"""Sensitivity analysis with an alternative p53 coding."""
import config_cox as cfg

import json
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

TAG = "p53alt"
SPACES = ["clinical", "radiomic_shape", "mixed_shape"]
ALTERNATIVE = ("assente", "intensa", "diffusa")


def main() -> int:
    out = cfg.RESULTS_DIR / "sensitivity_p53"
    out.mkdir(parents=True, exist_ok=True)

    from data_cox import load_dataset
    main_ds = load_dataset()
    original = cfg.P53_ABERRANT_CATEGORIES
    cfg.P53_ABERRANT_CATEGORIES = ALTERNATIVE
    ds = load_dataset()
    cfg.P53_ABERRANT_CATEGORIES = original

    a = main_ds.df["p53_ESPRESSIONE"].to_numpy()
    b = ds.df["p53_ESPRESSIONE"].to_numpy()
    changed = int((a != b).sum())
    print(f"\np53 wild-type: main {int((a == 1).sum())}, alternative {int((b == 1).sum())}; "
          f"patients recoded: {changed}")
    if changed == 0:
        sys.exit("STOP: no patient recoded - check the category names.")

    import checks_cox as ck
    from evaluation_cox import likelihood_ratio_rad_score, repeated_cv, summarise

    folds = repeated_cv(ds, SPACES, tag=TAG)
    summary = summarise(folds)
    main_path = cfg.RESULTS_DIR / "cox_cv_summary.csv"
    if main_path.exists():
        m = pd.read_csv(main_path)[["space", "c_test_mean"]].rename(
            columns={"c_test_mean": "c_cv_main"})
        summary = summary.merge(m, on="space", how="left")
        summary["difference"] = summary["c_test_mean"] - summary["c_cv_main"]
    summary.to_csv(out / "cv_summary.csv", index=False, encoding="utf-8-sig")
    print("\n--- Cross-validated C-index ---")
    cols = [c for c in ("space", "c_test_mean", "c_test_sd", "c_cv_main", "difference")
            if c in summary.columns]
    print(summary[cols].round(4).to_string(index=False))

    lr = likelihood_ratio_rad_score(ds, ck.oof_scores(ds, "radiomic_shape", TAG))
    lr = {k: v for k, v in lr.items() if k != "model_combined"}
    with open(out / "likelihood_ratio_oof.json", "w", encoding="utf-8") as fh:
        json.dump(lr, fh, indent=2, default=str)
    print(f"\nComplementarity: rad-score added chi2 = {lr['lr_score_added']:.2f} "
          f"(p = {lr['p_score_added']:.4g}); clinical added chi2 = "
          f"{lr['lr_clinical_added']:.2f} (p = {lr['p_clinical_added']:.4g})")

    windows = ck.landmark(ds, SPACES, tag=TAG)
    windows.to_csv(out / "windows.csv", index=False, encoding="utf-8-sig")
    print("\n--- C-index by window ---")
    print(windows.pivot_table(index=["order", "population", "n", "events"],
                              columns="Model", values="c_index")
          .reset_index().drop(columns="order").round(3).to_string(index=False))

    crossover = ck.crossover_test(ds, SPACES, tag=TAG)
    crossover.to_csv(out / "crossover.csv", index=False, encoding="utf-8-sig")
    print("\n--- Crossover ---")
    print(crossover[["horizon_months", "crossover", "ci_lower", "ci_upper",
                     "bootstrap_p"]].round(4).to_string(index=False))

    inter = ck.time_interaction(ds, tag=TAG)
    inter.to_csv(out / "time_interaction.csv", index=False, encoding="utf-8-sig")
    print("\n--- Time interaction ---")
    print(inter.to_string(index=False))
    print(f"\nsaved in: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
