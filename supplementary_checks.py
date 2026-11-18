"""Potential follow-up analysis, proportional hazards test with the out-of-fold score, and expected precision by cohort size."""
import config_cox as cfg

import shutil
import sys
import types
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def main() -> int:
    import checks_cox as ck
    import figures_cox as fg
    from data_cox import load_dataset

    out = cfg.RESULTS_DIR / "supplementary_checks"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    d, e = np.asarray(ds.duration, float), np.asarray(ds.event, int)
    spaces = list(cfg.FEATURE_SPACES)
    rad = next(s for s in spaces if s.startswith("radiomic"))

    lock = pd.Timestamp(cfg.DATA_LOCK)
    diag = pd.to_datetime(ds.df["date_diagnosis"]).reset_index(drop=True)
    potential = (lock - diag).dt.days / 30.4375
    print("\n=== A. Potential follow-up (data lock minus diagnosis) ===")
    print(f"minimum {potential.min():.2f} months; median {potential.median():.1f}; "
          f"< 6 months: {(potential < 6).sum()}; < 12 months: {(potential < 12).sum()}")
    print(f"censored before 6 months: {int(((d < 6) & (e == 0)).sum())}; "
          f"before 12 months: {int(((d < 12) & (e == 0)).sum())}")

    keep = (potential >= 12).to_numpy()
    print(f"\nSubset with >= 12 months potential follow-up: {int(keep.sum())} patients, "
          f"{int(e[keep].sum())} deaths (excluded {int((~keep).sum())})")
    full_oof = ck.oof_scores
    ck.oof_scores = lambda _ds, space, tag="main": full_oof(ds, space, tag)[keep]
    sub = types.SimpleNamespace(df=ds.df[keep].reset_index(drop=True),
                                duration=d[keep], event=e[keep],
                                clinical_features=list(ds.clinical_features))
    try:
        windows = ck.landmark(sub, spaces)
        windows.to_csv(out / "followup12_windows.csv", index=False, encoding="utf-8-sig")
        print(windows.pivot_table(index=["order", "population", "n", "events"],
                                  columns="Model", values="c_index")
              .reset_index().drop(columns="order").round(3).to_string(index=False))
        crossover = ck.crossover_test(sub, spaces)
        crossover.to_csv(out / "followup12_crossover.csv", index=False, encoding="utf-8-sig")
        print(crossover[["horizon_months", "advantage_early", "advantage_survivors",
                         "crossover", "ci_lower", "ci_upper", "bootstrap_p"]]
              .round(4).to_string(index=False))
        inter = ck.time_interaction(sub)
        inter.to_csv(out / "followup12_time_interaction.csv", index=False, encoding="utf-8-sig")
        print(inter.to_string(index=False))
    finally:
        ck.oof_scores = full_oof

    print("\n=== B. Scaled Schoenfeld test with the OUT-OF-FOLD rad-score ===")
    ph = ck.proportional_hazards(ck.build_frame(ds, ck.oof_scores(ds, rad)))
    ph.to_csv(out / "schoenfeld_oof.csv", index=False, encoding="utf-8-sig")
    main_ph = cfg.RESULTS_DIR / "cox_proportional_hazards.csv"
    if main_ph.exists():
        m = pd.read_csv(main_ph, encoding="utf-8-sig")[["variable", "p"]]
        ph = ph.merge(m.rename(columns={"p": "p_final_model_score"}), on="variable", how="left")
    print(ph.round(4).to_string(index=False))

    print("\n=== C. Expected precision by cohort size, n = 28 added ===")
    old = cfg.RESULTS_DIR / "cox_sample_size.csv"
    if old.exists():
        shutil.copy2(old, cfg.RESULTS_DIR / "cox_sample_size_old.csv")
    table = ck.sample_size(ds, spaces, sizes=(28, 40, 60, 80, 100, 150, 200, 300))
    table.to_csv(old, index=False, encoding="utf-8-sig")
    print(table.round(3).to_string(index=False))
    fg.plot_sample_size(table)
    print(f"\nsaved in: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
