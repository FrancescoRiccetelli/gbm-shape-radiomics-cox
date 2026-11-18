"""Table 1: characteristics of the derivation and external cohorts."""
import config_cox as cfg

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CLINICAL = list(cfg.CLINICAL_FEATURES)


def _fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _median_iqr(v: pd.Series) -> str:
    v = v.dropna()
    if v.empty:
        return "NA"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    out = f"{med:.1f} ({q1:.1f}-{q3:.1f})"
    return out


def _count(mask: pd.Series, valid: pd.Series) -> str:
    n_valid = int(valid.sum())
    if n_valid == 0:
        return "NA"
    k = int((mask & valid).sum())
    return f"{k} ({100 * k / n_valid:.0f}%)"


def _missing(v: pd.Series) -> int:
    return int(v.isna().sum())


def _km(duration, event):
    from lifelines import KaplanMeierFitter
    from lifelines.utils import median_survival_times

    km = KaplanMeierFitter().fit(duration, event)
    median = km.median_survival_time_
    ci = median_survival_times(km.confidence_interval_)
    lo, hi = ci.iloc[0, 0], ci.iloc[0, 1]
    surv = {h: float(km.predict(h)) for h in (6.0, 12.0)}
    rev = KaplanMeierFitter().fit(duration, 1 - np.asarray(event))
    follow_up = rev.median_survival_time_
    return median, lo, hi, surv, follow_up


def _fmt_months(x) -> str:
    return "not reached" if x is None or np.isinf(x) or np.isnan(x) else f"{x:.1f}"


def describe(df: pd.DataFrame, field) -> dict:
    d, e = df[cfg.DURATION_COL].to_numpy(float), df[cfg.EVENT_COL].to_numpy(int)
    median, lo, hi, surv, follow_up = _km(d, e)
    out = {
        "Patients, n": str(len(df)),
        "Age, years, median (IQR)": _median_iqr(df["Eta"]),
        "Male sex, n (%)": _count(df["GENERE"] == 0, df["GENERE"].notna()),
        "Gross total resection, n (%)": _count(df["RESEZIONE"] == 1,
                                               df["RESEZIONE"].notna()),
        "MGMT methylated (>=10%), n (%)": _count(df["MGMT_METILAZIONE"] == 1,
                                                 df["MGMT_METILAZIONE"].notna()),
        "p53 aberrant, n (%)": _count(df["p53_ESPRESSIONE"] == 0,
                                      df["p53_ESPRESSIONE"].notna()),
        "Ki67, %, median (IQR)": _median_iqr(df["Ki67"]),
        "Deaths, n (%)": f"{int(e.sum())} ({100 * e.mean():.0f}%)",
        "Deaths within 6 months, n (%)":
            f"{int(((d < 6) & (e == 1)).sum())} ({100 * ((d < 6) & (e == 1)).mean():.0f}%)",
        "Deaths within 12 months, n (%)":
            f"{int(((d < 12) & (e == 1)).sum())} ({100 * ((d < 12) & (e == 1)).mean():.0f}%)",
        "OS at 6 months, % (KM)": f"{100 * surv[6.0]:.0f}",
        "OS at 12 months, % (KM)": f"{100 * surv[12.0]:.0f}",
        "Median OS, months (95% CI)":
            f"{_fmt_months(median)} ({_fmt_months(lo)}-{_fmt_months(hi)})",
        "Median follow-up, months (reverse KM)": _fmt_months(follow_up),
    }
    if field is not None:
        out["MRI 1.5 T, n (%)"] = f"{int((field == 1.5).sum())} ({100 * (field == 1.5).mean():.0f}%)"
        out["MRI 3 T, n (%)"] = f"{int((field == 3.0).sum())} ({100 * (field == 3.0).mean():.0f}%)"
    else:
        out["MRI 1.5 T, n (%)"] = "NA"
        out["MRI 3 T, n (%)"] = "NA"
    return out


def p_values(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    from lifelines.statistics import logrank_test
    from scipy.stats import fisher_exact, mannwhitneyu

    def mw(col):
        x, y = a[col].dropna(), b[col].dropna()
        return mannwhitneyu(x, y).pvalue if len(x) and len(y) else np.nan

    def fisher(col, value):
        x, y = a[col].dropna(), b[col].dropna()
        if not len(x) or not len(y):
            return np.nan
        table = [[int((x == value).sum()), int((x != value).sum())],
                 [int((y == value).sum()), int((y != value).sum())]]
        return fisher_exact(table)[1]

    def fisher_mask(ma, mb):
        table = [[int(ma.sum()), int((~ma).sum())], [int(mb.sum()), int((~mb).sum())]]
        return fisher_exact(table)[1]

    da, ea = a[cfg.DURATION_COL], a[cfg.EVENT_COL]
    db, eb = b[cfg.DURATION_COL], b[cfg.EVENT_COL]
    return {
        "Age, years, median (IQR)": mw("Eta"),
        "Male sex, n (%)": fisher("GENERE", 0),
        "Gross total resection, n (%)": fisher("RESEZIONE", 1),
        "MGMT methylated (>=10%), n (%)": fisher("MGMT_METILAZIONE", 1),
        "p53 aberrant, n (%)": fisher("p53_ESPRESSIONE", 0),
        "Ki67, %, median (IQR)": mw("Ki67"),
        "Deaths within 6 months, n (%)":
            fisher_mask(((da < 6) & (ea == 1)).to_numpy(), ((db < 6) & (eb == 1)).to_numpy()),
        "Deaths within 12 months, n (%)":
            fisher_mask(((da < 12) & (ea == 1)).to_numpy(), ((db < 12) & (eb == 1)).to_numpy()),
        "Median OS, months (95% CI)":
            logrank_test(da, db, ea, eb).p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True, help="Rome clinical file")
    parser.add_argument("--radiomic", required=True, help="Rome radiomic file")
    parser.add_argument("--data-lock", default=cfg.DATA_LOCK)
    args = parser.parse_args()

    from data_cox import field_strength, load_dataset
    from validate_external import MODEL_PATH, load_external

    ds = load_dataset()
    cremona = ds.df.copy()
    field = field_strength(cremona[cfg.ID_COL])

    frozen = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    rome_all = load_external(args.clinical, args.radiomic, args.data_lock, frozen)
    complete = rome_all[CLINICAL].notna().all(axis=1)
    rome = rome_all[complete].reset_index(drop=True)

    columns = {
        f"Cremona (n = {len(cremona)})": describe(cremona, field),
        f"Rome (n = {len(rome)})": describe(rome, None),
    }
    pvals = p_values(cremona, rome)

    rows = [k for k in next(iter(columns.values())) ]
    table = pd.DataFrame({name: [col[r] for r in rows] for name, col in columns.items()},
                         index=rows)
    table["p"] = [_fmt_p(pvals.get(r, np.nan)) for r in rows]
    table.index.name = "Characteristic"

    out = cfg.RESULTS_DIR / "table1.csv"
    table.to_csv(out, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print("\n" + table.to_string())
    print(f"\nRome: {len(rome_all)} patients with imaging, {len(rome)} with complete "
          f"clinical data (excluded: {len(rome_all) - len(rome)})")
    print("\nNotes: time zero is histological diagnosis in Cremona and surgery in Rome;"
          " p values are descriptive (Mann-Whitney, Fisher, log-rank).")
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
