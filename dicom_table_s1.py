"""Supplementary Table S1: MRI acquisition parameters read from DICOM headers."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import pydicom
except ImportError:
    sys.exit("pydicom is missing: py -m pip install pydicom")

T1_KEYS = ["T1", "MPRAGE", "BRAVO", "SPGR", "TFE", "VIBE", "THRIVE", "BRAINVIEW",
           "NEURONAVIGATORE", "3D_MC", "S3DI_MC"]
POST_KEYS = ["POST", "GD", "GAD", "+C", "C+", "MDC", "CONTR", " CE", "_CE", "DOPO",
             "DOTAREM", "PROHANCE", "CLARICYCLIC", "GADOVIST", "MULTIHANCE",
             "ML ", "ML_"]
FLAIR_KEYS = ["FLAIR", "DARK FLUID", "DARK-FLUID", "DARKFLUID"]
EXCLUDE_KEYS = ["CBV", "CBF", "MTT", "PERFUS", "DSC", "DTI", "ADC", "TRACE",
                "REGISTERED", " ON T1", "+ T1", "+T1", "RGB", "FUSION",
                "MAP", "SUB", "SURVEY", "LOCALI", "MIP"]
MAX_SLICE_MM = 10.0

FIELDS = {
    "manufacturer": ("Manufacturer", None),
    "model": ("ManufacturerModelName", None),
    "field_T": ("MagneticFieldStrength", float),
    "TR_ms": ("RepetitionTime", float),
    "TE_ms": ("EchoTime", float),
    "TI_ms": ("InversionTime", float),
    "flip_deg": ("FlipAngle", float),
    "slice_mm": ("SliceThickness", float),
    "rows": ("Rows", int),
    "cols": ("Columns", int),
    "acq_type": ("MRAcquisitionType", None),
    "contrast_agent": ("ContrastBolusAgent", None),
    "series_description": ("SeriesDescription", None),
    "protocol": ("ProtocolName", None),
}


def value(ds, key, cast):
    v = ds.get(key)
    if v is None or v == "":
        return None
    try:
        return cast(v) if cast else str(v).strip()
    except (TypeError, ValueError):
        return None


def classify(row, t1_keys, post_keys, flair_keys) -> str:
    desc = f" {row['series_description'] or ''} ".upper()
    text = f" {row['series_description'] or ''} {row['protocol'] or ''} ".upper()
    reformat = re.search(r"\bMPR\b", desc) is not None
    if reformat or any(k in desc for k in EXCLUDE_KEYS):
        return "other"
    if row["slice_mm"] is not None and row["slice_mm"] > MAX_SLICE_MM:
        return "other"
    ti = row["TI_ms"]
    if any(k in text for k in flair_keys) or (ti is not None and ti > 1500):
        return "FLAIR"
    is_t1 = any(k in text for k in t1_keys)
    if not is_t1:
        return "other"
    te = row["TE_ms"]
    if te is not None and te > 30:
        return "other"
    has_agent = bool((row["contrast_agent"] or "").strip())
    if any(k in text for k in post_keys) or has_agent:
        return "T1-CE"
    return "other"


def read_headers(root: Path) -> pd.DataFrame:
    seen, rows, n_files = set(), [], 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        n_files += 1
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = ds.get("SeriesInstanceUID")
        if uid is None or ds.get("Modality") != "MR" or uid in seen:
            continue
        seen.add(uid)
        try:
            study_folder = path.relative_to(root).parts[0]
        except (ValueError, IndexError):
            study_folder = ""
        study_folder = study_folder.lstrip("0") or study_folder
        image_type = [str(x).upper() for x in (ds.get("ImageType") or [])]
        row = {"study_id": study_folder, "series_uid": str(uid),
               "derived": "DERIVED" in image_type}
        for name, (key, cast) in FIELDS.items():
            row[name] = value(ds, key, cast)
        for k in ("TR_ms", "TE_ms", "TI_ms"):
            if row[k] is not None and row[k] <= 0:
                row[k] = None
        ps = ds.get("PixelSpacing")
        row["pixel_mm"] = float(ps[0]) if ps is not None and len(ps) == 2 else None
        rows.append(row)
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} series read ({n_files} files scanned)", flush=True)
    print(f"  {len(rows)} MR series in {n_files} files")
    return pd.DataFrame(rows)


def summary(values: pd.Series, fmt: str) -> str:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return "NA"
    med, lo, hi = v.median(), v.min(), v.max()
    if abs(hi - lo) < 1e-9:
        return format(med, fmt)
    return f"{format(med, fmt)} ({format(lo, fmt)}-{format(hi, fmt)})"


def most_common(values: pd.Series) -> str:
    v = values.dropna().astype(str).str.strip()
    v = v[v != ""]
    if v.empty:
        return "NA"
    return "; ".join(f"{k} ({n})" for k, n in v.value_counts().head(3).items())


def pick_series(series: pd.DataFrame) -> pd.DataFrame:
    """One series per patient and sequence: 3D first, then thinnest slices."""
    usable = series[series["sequence"] != "other"].copy()
    usable["is_3d"] = usable["acq_type"].fillna("").str.upper().eq("3D")
    usable = usable.sort_values(["study_id", "sequence", "is_3d", "slice_mm"],
                                ascending=[True, True, False, True])
    return usable.drop_duplicates(["study_id", "sequence"])


def scanner_label(df: pd.DataFrame) -> pd.Series:
    field = df["field_T"].map(lambda f: f"{f:.1f} T" if pd.notna(f) else "field n/a")
    return (df["manufacturer"].fillna("vendor n/a") + " "
            + df["model"].fillna("model n/a") + ", " + field)


def table(chosen: pd.DataFrame, group: str = "scanner") -> pd.DataFrame:
    chosen = chosen.assign(scanner=scanner_label(chosen))
    if group == "field":
        chosen["group"] = chosen["field_T"].map(
            lambda f: f"{f:.1f} T" if pd.notna(f) else "field not recorded")
    else:
        chosen["group"] = chosen["scanner"]
    out = {}
    for name, block in chosen.groupby("group"):
        col = {"Patients, n": str(block["study_id"].nunique())}
        if group == "field":
            per = block.drop_duplicates(["study_id", "scanner"])["scanner"].value_counts()
            col["Scanners (patients)"] = "; ".join(
                f"{k.rsplit(',', 1)[0]} ({n})" for k, n in per.items())
        for seq in ("T1-CE", "FLAIR"):
            b = block[block["sequence"] == seq]
            if b.empty:
                continue
            col[f"{seq}: series, n"] = str(len(b))
            col[f"{seq}: 2D/3D"] = most_common(b["acq_type"])
            col[f"{seq}: TR (ms)"] = summary(b["TR_ms"], ".1f")
            col[f"{seq}: TE (ms)"] = summary(b["TE_ms"], ".1f")
            if seq == "FLAIR":
                col[f"{seq}: TI (ms)"] = summary(b["TI_ms"], ".0f")
            col[f"{seq}: flip angle (deg)"] = summary(b["flip_deg"], ".0f")
            col[f"{seq}: slice thickness (mm)"] = summary(b["slice_mm"], ".1f")
            col[f"{seq}: in-plane pixel (mm)"] = summary(b["pixel_mm"], ".2f")
            col[f"{seq}: matrix (rows)"] = summary(b["rows"], ".0f")
        col["Contrast agent (DICOM)"] = most_common(
            block.loc[block["sequence"] == "T1-CE", "contrast_agent"])
        out[name] = col
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", required=True, help="folder with one subfolder per patient")
    parser.add_argument("--label", default="Derivation")
    parser.add_argument("--cohort", default=None,
                        help="CSV with columns study_id and field_T_used (export_cohort_ids.py)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--t1ce", default="", help="extra T1 sequence keywords")
    parser.add_argument("--post", default="", help="extra contrast keywords")
    parser.add_argument("--flair", default="", help="extra FLAIR keywords")
    parser.add_argument("--group", choices=["scanner", "field"], default="scanner",
                        help="one column per scanner (default) or per field strength")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"{root} is not a folder")
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "results_cox" / "dicom_s1"
    out.mkdir(parents=True, exist_ok=True)

    extra = lambda s: [k.strip().upper() for k in s.split(",") if k.strip()]
    t1_keys = T1_KEYS + extra(args.t1ce)
    post_keys = POST_KEYS + extra(args.post)
    flair_keys = FLAIR_KEYS + extra(args.flair)

    print(f"Reading DICOM headers under {root} ...")
    series = read_headers(root)
    if series.empty:
        sys.exit("No MR DICOM series found.")
    series["sequence"] = [classify(r, t1_keys, post_keys, flair_keys)
                          for _, r in series.iterrows()]

    cohort = None
    if args.cohort:
        cohort = pd.read_csv(args.cohort, dtype={"study_id": str})
        cohort["study_id"] = cohort["study_id"].str.strip().str.lstrip("0")
        ids = set(cohort["study_id"])
        found = set(series["study_id"])
        print(f"\nCohort: {len(ids)} patients; with DICOM folder: {len(ids & found)}; "
              f"folders not in the cohort (excluded): {len(found - ids)}")
        missing = sorted(ids - found, key=lambda s: (len(s), s))
        if missing:
            print(f"  cohort patients without a DICOM folder: {', '.join(missing)}")
        series = series[series["study_id"].isin(ids)]

    series.to_csv(out / f"{args.label}_series.csv", index=False, encoding="utf-8-sig")

    fields = series.groupby("study_id")["field_T"].agg(lambda v: sorted(set(v.dropna())))
    single = {k: v[0] for k, v in fields.items() if len(v) == 1}
    missing = series["field_T"].isna()
    series.loc[missing, "field_T"] = series.loc[missing, "study_id"].map(single)

    kinds = fields.map(lambda v: "1.5 T only" if v == [1.5] else "3.0 T only" if v == [3.0]
                       else "both" if len(v) > 1 else "not recorded")
    print("\nPatients by field strength in their DICOM series:")
    print(kinds.value_counts().to_string())
    both = sorted(kinds[kinds == "both"].index, key=lambda s: (len(s), s))
    if both:
        print(f"  with series at both fields: {', '.join(both)}")

    chosen = pick_series(series)
    n_pat = series["study_id"].nunique()
    for seq in ("T1-CE", "FLAIR"):
        have = chosen.loc[chosen["sequence"] == seq, "study_id"].nunique()
        print(f"Patients with {seq}: {have} of {n_pat}")
        lacking = sorted(set(series["study_id"]) -
                         set(chosen.loc[chosen["sequence"] == seq, "study_id"]),
                         key=lambda s: (len(s), s))
        if lacking:
            print(f"  without {seq}: {', '.join(lacking)}")

    print("\nT1-like series left as 'other' (check for missed post-contrast T1):")
    t1_other = series[(series["sequence"] == "other") &
                      series["series_description"].fillna("").str.upper()
                      .apply(lambda t: any(k in t for k in t1_keys))]
    print(t1_other["series_description"].value_counts().head(25).to_string()
          if not t1_other.empty else "  none")

    if cohort is not None and "field_T_used" in cohort.columns:
        dicom_field = (series.groupby("study_id")["field_T"]
                       .agg(lambda v: "/".join(sorted({f"{x:.1f}" for x in v.dropna()}))))
        dicom_model = (series.groupby("study_id")
                       .apply(lambda b: "; ".join(sorted(set(scanner_label(b))))))
        check = cohort.set_index("study_id")[["field_T_used"]].join(
            dicom_field.rename("field_T_dicom"), how="left").join(
            dicom_model.rename("scanners_dicom"), how="left")
        check["match"] = [
            (str(d) == f"{u:.1f}") if isinstance(d, str) and d else None
            for u, d in zip(check["field_T_used"], check["field_T_dicom"])]
        check.to_csv(out / f"{args.label}_field_check.csv", encoding="utf-8-sig")
        n_ok = int((check["match"] == True).sum())
        n_bad = int((check["match"] == False).sum())
        print(f"\nField strength, DICOM vs used for harmonisation: "
              f"{n_ok} match, {n_bad} differ, {int(check['match'].isna().sum())} not checkable")
        if n_bad:
            print(check[check["match"] == False].to_string())

    s1 = table(chosen, args.group)
    suffix = "" if args.group == "scanner" else "_by_field"
    s1.to_csv(out / f"{args.label}_table_s1{suffix}.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + s1.to_string())
    print(f"\nsaved in {out}")
    print("Share only the _table_s1.csv and _field_check.csv files: "
          "_series.csv may contain names in the series descriptions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
