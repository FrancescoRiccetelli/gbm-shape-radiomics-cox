"""Comparison of the out-of-fold radiomic score with tumour volume alone."""
import config_cox as cfg

import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HORIZONS = (6.0, 12.0)
N_BOOT = 2000


def cindex(d, e, s):
    from sksurv.metrics import concordance_index_censored
    e = np.asarray(e).astype(bool)
    if e.sum() == 0:
        return np.nan
    return float(concordance_index_censored(e, np.asarray(d, float),
                                            np.asarray(s, float))[0])


def windows(d, e):
    """Label, duration, event and mask for every follow-up window of the main analysis."""
    out = []
    for t in HORIZONS:
        ev = (d < t) & (e == 1)
        out.append((f"First {t:.0f} months", np.minimum(d, t), ev, np.ones(len(d), bool)))
    out.append(("Whole follow-up", d, e == 1, np.ones(len(d), bool)))
    for t in HORIZONS:
        alive = d >= t
        out.append((f"Alive at {t:.0f} months", d - t, e == 1, alive))
    return out


def main() -> int:
    import checks_cox as ck
    from data_cox import load_dataset

    out_dir = cfg.RESULTS_DIR / "volume_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    original = cfg.SCANNER_CORRECTION
    cfg.SCANNER_CORRECTION = "none"
    try:
        raw = load_dataset()
    finally:
        cfg.SCANNER_CORRECTION = original
    if not (raw.df[cfg.ID_COL].to_numpy() == ds.df[cfg.ID_COL].to_numpy()).all():
        sys.exit("Patient order differs between datasets: stopping.")

    vol = cfg.VOLUME_FEATURE
    core = raw.df[f"enh_{vol}"].to_numpy(float) + raw.df[f"nec_{vol}"].to_numpy(float)
    total = raw.df["volume_total"].to_numpy(float)
    rad = ck.oof_scores(ds, "radiomic_shape")
    d, e = np.asarray(ds.duration, float), np.asarray(ds.event, int)

    scores = {"rad-score (out-of-fold)": rad, "core volume": core,
              "total volume": total}
    print(f"\nPatients: {len(d)}, deaths: {e.sum()}")
    print(f"Core volume, median (IQR): {np.median(core):.1f} "
          f"({np.percentile(core, 25):.1f}-{np.percentile(core, 75):.1f})")

    rng = np.random.default_rng(cfg.SEED)
    boot_idx = [rng.integers(0, len(d), len(d)) for _ in range(N_BOOT)]
    rows = []
    for label, dd, ee, mask in windows(d, e):
        ddm, eem = dd[mask], ee[mask]
        observed = {k: cindex(ddm, eem, s[mask]) for k, s in scores.items()}
        boot = {k: [] for k in scores}
        diff = []
        for idx in boot_idx:
            pick = idx[mask[idx]] if not mask.all() else idx
            if len(pick) < 10:
                continue
            values = {k: cindex(dd[pick], ee[pick], s[pick]) for k, s in scores.items()}
            for k, v in values.items():
                if not np.isnan(v):
                    boot[k].append(v)
            a, b = values["rad-score (out-of-fold)"], values["core volume"]
            if not (np.isnan(a) or np.isnan(b)):
                diff.append(a - b)
        row = {"window": label, "patients": int(mask.sum()), "deaths": int(eem.sum())}
        for k in scores:
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            row[k] = round(observed[k], 3)
            row[f"{k} 95% CI"] = f"{lo:.3f}-{hi:.3f}"
        diff = np.array(diff)
        point = observed["rad-score (out-of-fold)"] - observed["core volume"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
        row["difference rad-score minus core volume"] = round(point, 3)
        row["difference 95% CI"] = f"{lo:+.3f} to {hi:+.3f}"
        row["difference bootstrap p"] = round(max(p, 1 / len(diff)), 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "windows.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n--- C-index by window ---")
    print(table.to_string(index=False))

    from lifelines import CoxPHFitter
    from scipy.stats import chi2, spearmanr

    def loglik(cols):
        frame = pd.DataFrame(cols)
        frame["duration"], frame["event"] = d, e
        return CoxPHFitter().fit(frame, "duration", "event").log_likelihood_

    lr = {}
    for name, v in (("core", core), ("total", total)):
        logv = np.log(v + 1.0)
        ll_vol = loglik({"log_volume": logv})
        ll_rad = loglik({"rad_score": rad})
        ll_both = loglik({"log_volume": logv, "rad_score": rad})
        a, b = 2 * (ll_both - ll_vol), 2 * (ll_both - ll_rad)
        lr[name] = {"rad_score_added_to_volume_chi2": round(a, 2),
                    "rad_score_added_to_volume_p": float(chi2.sf(a, 1)),
                    "volume_added_to_rad_score_chi2": round(b, 2),
                    "volume_added_to_rad_score_p": float(chi2.sf(b, 1))}
    with open(out_dir / "likelihood_ratio.json", "w", encoding="utf-8") as fh:
        json.dump(lr, fh, indent=2)
    print("\n--- Likelihood ratio tests (1 df each) ---")
    for name, r in lr.items():
        print(f"  {name} volume: rad-score added chi2 = "
              f"{r['rad_score_added_to_volume_chi2']:.2f} (p = {r['rad_score_added_to_volume_p']:.4g}); "
              f"volume added chi2 = {r['volume_added_to_rad_score_chi2']:.2f} "
              f"(p = {r['volume_added_to_rad_score_p']:.4g})")

    corr = {name: {"rho": round(float(spearmanr(rad, v)[0]), 3),
                   "p": float(spearmanr(rad, v)[1])}
            for name, v in (("core", core), ("total", total))}
    with open(out_dir / "correlation.json", "w", encoding="utf-8") as fh:
        json.dump(corr, fh, indent=2)
    print("\n--- Spearman correlation with rad-score ---")
    for name, c in corr.items():
        print(f"  {name} volume: rho = {c['rho']:.3f} (p = {c['p']:.3g})")

    print(f"\nsaved in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
