"""Sensitivity analysis of the external validation without harmonising the external features."""
import config_cox as cfg

import json
import sys

import pandas as pd


def main() -> int:
    import validate_external as ve

    out = cfg.RESULTS_DIR / "external_validation_raw"
    out.mkdir(parents=True, exist_ok=True)
    ve.OUT_DIR = out
    ve.harmonise = lambda frame, frozen: frame

    code = ve.main()

    rows = []
    for space in ("clinical", "radiomic_shape", "mixed_shape"):
        for tag, folder in (("harmonised", cfg.RESULTS_DIR / "external_validation"),
                            ("raw", out)):
            path = folder / f"validation_{space}_common.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            high = sum(g["n"] for k, g in r.get("risk_groups", {}).items()
                       if k.lower().startswith("high"))
            rows.append({"model": cfg.SPACE_LABELS.get(space, space), "external features": tag,
                         "C-index": round(r["c_index"], 3),
                         "C 95% CI": f"{r['c_index_ci'][0]:.3f}-{r['c_index_ci'][1]:.3f}",
                         "expected C": round(r["c_expected"], 3),
                         "HR per SD": round(r["hr_per_sd"], 2),
                         "p (HR)": round(r["hr_per_sd_p"], 3),
                         "calibration slope": round(r["calibration_slope"], 2),
                         "high-risk": f"{high}/{r['n']}",
                         "log-rank p": (None if r["logrank_p"] is None
                                        or r["logrank_p"] != r["logrank_p"]
                                        else round(r["logrank_p"], 3))})
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison_harmonised_vs_raw.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n=== 28 patients with complete data: harmonised vs raw external features ===")
    print(table.to_string(index=False))
    print(f"\nsaved: {out / 'comparison_harmonised_vs_raw.csv'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
