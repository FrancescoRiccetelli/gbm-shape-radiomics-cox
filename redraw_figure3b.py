"""Figure 3B: hazard ratio of MGMT before and after 12 months from the time-interaction model."""
import config_cox as cfg

import shutil
import sys

import numpy as np
import pandas as pd

FOCUS = "MGMT_METILAZIONE"
SPLIT = 12.0


def main() -> int:
    import checks_cox as ck
    import figures_cox as fg
    from data_cox import load_dataset
    from lifelines import CoxTimeVaryingFitter

    saved_path = cfg.RESULTS_DIR / "cox_time_interaction.csv"
    if not saved_path.exists():
        sys.exit(f"{saved_path} not found: run the pipeline first.")
    saved = pd.read_csv(saved_path, encoding="utf-8-sig")
    saved = saved[saved["term"] == "MGMT (adjusted)"]
    if saved.empty:
        sys.exit("Row 'MGMT (adjusted)' not found in cox_time_interaction.csv.")
    saved = saved.iloc[0]

    ds = load_dataset()
    rad = next(s for s in cfg.FEATURE_SPACES if s.startswith("radiomic"))
    X = ds.df[list(ds.clinical_features)].reset_index(drop=True).copy()
    score = ck.oof_scores(ds, rad, "main")
    X["rad_score"] = (score - score.mean()) / score.std()

    d, e = ds.duration, ds.event
    columns = list(X.columns)
    values = X.to_numpy(float)
    rows = []
    for i in range(len(d)):
        if d[i] <= SPLIT:
            rows.append((i, 0.0, d[i], e[i], 0, *values[i]))
        else:
            rows.append((i, 0.0, SPLIT, 0, 0, *values[i]))
            rows.append((i, SPLIT, d[i], e[i], 1, *values[i]))
    frame = pd.DataFrame(rows, columns=["id", "start", "stop", "event", "late"] + columns)
    keys = dict(id_col="id", start_col="start", stop_col="stop", event_col="event")

    out = []
    for late, label in ((0, f"0-{SPLIT:.0f} months"), (1, f">{SPLIT:.0f} months")):
        sub = frame[frame.late == late]
        r = CoxTimeVaryingFitter().fit(
            sub[["id", "start", "stop", "event"] + columns], **keys).summary.loc[FOCUS]
        out.append({"variable": FOCUS, "window": label,
                    "hazard_ratio": float(np.exp(r["coef"])),
                    "ci_lower": float(np.exp(r["coef lower 95%"])),
                    "ci_upper": float(np.exp(r["coef upper 95%"])),
                    "p": float(r["p"]), "events": int(sub.event.sum())})
    table = pd.DataFrame(out)

    expected = (float(saved[f"hr_0_{SPLIT:.0f}m"]), float(saved[f"hr_after_{SPLIT:.0f}m"]))
    got = tuple(table["hazard_ratio"])
    if any(abs(a - b) > 0.002 for a, b in zip(got, expected)):
        sys.exit(f"STOP: HR {got} differ from cox_time_interaction.csv {expected}.")

    table["p_interaction"] = float(saved["p_interaction"])
    table.round(4).to_csv(cfg.RESULTS_DIR / "cox_time_varying_mgmt_adjusted.csv",
                          index=False, encoding="utf-8-sig")
    print(table.round(3).to_string(index=False))
    print(f"p for interaction = {float(saved['p_interaction']):.3f}")

    old = cfg.FIGURES_DIR / f"fig_cox_time_varying.{cfg.FIG_FORMAT}"
    if old.exists():
        shutil.copy2(old, old.with_name(f"fig_cox_time_varying_old.{cfg.FIG_FORMAT}"))
    fg.plot_time_varying(table, panel="B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
