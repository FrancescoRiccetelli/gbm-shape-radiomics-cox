"""Sensitivity analysis excluding patients who died within the first months after diagnosis."""
import config_cox as cfg

import argparse
import sys
import types
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=float, default=2.0,
                        help="exclude patients who died before this time")
    args = parser.parse_args()

    import checks_cox as ck
    from data_cox import load_dataset

    ds = load_dataset()
    d, e = np.asarray(ds.duration, float), np.asarray(ds.event, int)
    keep = ~((d < args.months) & (e == 1))
    dropped = int((~keep).sum())
    print(f"\nExcluded: {dropped} patients who died within {args.months:g} months "
          f"of diagnosis; analysed: {int(keep.sum())} patients, "
          f"{int(e[keep].sum())} deaths")

    full_oof = ck.oof_scores
    ck.oof_scores = lambda _ds, space, tag="main": full_oof(ds, space, tag)[keep]

    sub = types.SimpleNamespace(
        df=ds.df[keep].reset_index(drop=True),
        duration=d[keep], event=e[keep],
        clinical_features=list(ds.clinical_features))

    spaces = list(cfg.FEATURE_SPACES)
    out = cfg.RESULTS_DIR / "sensitivity_early_deaths"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    windows = ck.landmark(sub, spaces)
    windows.to_csv(out / "windows.csv", index=False, encoding="utf-8-sig")
    print("\n--- C-index by window (out-of-fold) ---")
    print(windows.pivot_table(index=["order", "population", "n", "events"],
                              columns="Model", values="c_index")
          .reset_index().drop(columns="order").to_string(index=False))

    crossover = ck.crossover_test(sub, spaces)
    crossover.to_csv(out / "crossover.csv", index=False, encoding="utf-8-sig")
    print("\n--- Crossover ---")
    print(crossover[["horizon_months", "deaths_early_window", "advantage_early",
                     "advantage_survivors", "crossover", "ci_lower", "ci_upper",
                     "bootstrap_p"]].to_string(index=False))

    interaction = ck.time_interaction(sub)
    interaction.to_csv(out / "time_interaction.csv", index=False, encoding="utf-8-sig")
    print("\n--- Hazard ratios in the first 12 months and thereafter ---")
    print(interaction.to_string(index=False))

    print(f"\nsaved in: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
