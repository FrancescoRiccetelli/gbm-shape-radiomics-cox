"""Field-strength differences of the analysed shape features before and after harmonisation."""
import config_cox as cfg

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def main() -> int:
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    from data_cox import combat_adjust, field_strength, load_dataset

    original = cfg.SCANNER_CORRECTION
    cfg.SCANNER_CORRECTION = "none"
    try:
        ds = load_dataset()
    finally:
        cfg.SCANNER_CORRECTION = original

    columns = ds.columns_for("radiomic_shape")
    field = field_strength(ds.df[cfg.ID_COL])
    print(f"\nFeatures analysed: {len(columns)}; "
          f"1.5T n = {(field == 1.5).sum()}, 3T n = {(field == 3.0).sum()}")

    region_columns = [c for c in ds.df.columns
                      if any(c.startswith(f"{p}_") for p in cfg.REGIONS)]
    harmonised = ds.df.copy()
    harmonised[region_columns] = combat_adjust(ds.df[region_columns], field)

    def test(frame):
        p = np.array([mannwhitneyu(frame.loc[field == 1.5, c],
                                   frame.loc[field == 3.0, c]).pvalue
                      for c in columns])
        return p, multipletests(p, method="fdr_bh")[1]

    p_before, q_before = test(ds.df)
    p_after, q_after = test(harmonised)

    table = pd.DataFrame({
        "feature": columns,
        "harmonised_by_combat": [c in region_columns for c in columns],
        "p_before": p_before, "p_fdr_before": q_before,
        "p_after": p_after, "p_fdr_after": q_after,
    })
    table.to_csv(cfg.RESULTS_DIR / "scanner_check_shape.csv", index=False,
                 encoding="utf-8-sig")

    print(f"Differing by field strength (FDR < 0.05), before ComBat: "
          f"{int((q_before < 0.05).sum())}/{len(columns)}")
    print(f"Differing by field strength (FDR < 0.05), after ComBat : "
          f"{int((q_after < 0.05).sum())}/{len(columns)}")
    not_harm = table.loc[~table["harmonised_by_combat"], "feature"].tolist()
    if not_harm:
        print(f"Not harmonised (computed from raw volumes): {not_harm}")
    still = table.loc[table["p_fdr_after"] < 0.05, "feature"].tolist()
    if still:
        print(f"Still differing after ComBat: {still}")
    print(f"\nsaved: {cfg.RESULTS_DIR / 'scanner_check_shape.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
