"""Loading, recoding and scanner harmonisation of the clinical and radiomic data."""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

import config_cox as cfg


def _to_number(series: pd.Series) -> pd.Series:
    """Parse Italian-formatted numeric strings ('4,00%', '53,0054') to float."""
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _read_ods(path) -> pd.DataFrame:
    """Read the first sheet of a LibreOffice spreadsheet."""
    from odf.opendocument import load
    from odf.table import Table, TableCell, TableRow
    from odf.text import P

    sheet = load(str(path)).spreadsheet.getElementsByType(Table)[0]
    rows = []
    for row in sheet.getElementsByType(TableRow):
        cells = []
        for cell in row.getElementsByType(TableCell):
            repeat = int(cell.getAttribute("numbercolumnsrepeated") or 1)
            text = "".join(str(p) for p in cell.getElementsByType(P))
            cells += [text] * min(repeat, 64)
        rows.append(cells)

    header = [h.strip() for h in rows[0]]
    width = max(i for i, h in enumerate(header) if h) + 1
    header = header[:width]
    body = [r[:width] + [""] * (width - len(r[:width])) for r in rows[1:]]
    frame = pd.DataFrame(body, columns=header)
    frame = frame[frame[header[0]].str.strip() != ""]
    return frame.replace("", np.nan).reset_index(drop=True)


def _read_clinical_csv(path) -> pd.DataFrame:
    """Read the clinical CSV export, tolerating rows wrapped in quotes."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = next(csv.reader([lines[0]]))

    rows = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].replace('""', '"')
        rows.append(next(csv.reader([line])))

    frame = pd.DataFrame(rows, columns=header)
    if frame.shape[1] < 5:
        raise ValueError(f"Clinical file parsed into {frame.shape[1]} columns")
    return frame


def load_clinical() -> pd.DataFrame:
    """Clinical data with the survival endpoint attached."""
    path = cfg.DATA_CLINICAL
    df = (_read_ods(path) if str(path).lower().endswith(".ods")
          else _read_clinical_csv(path))

    df[cfg.ID_COL] = pd.to_numeric(df[cfg.ID_COL], errors="coerce")
    df = df[df[cfg.ID_COL].notna()].copy()
    df[cfg.ID_COL] = df[cfg.ID_COL].astype(int)

    df["date_diagnosis"] = pd.to_datetime(
        df["DATA_DIAGNOSI_ISTOLOGICA"], format="%d/%m/%Y")
    df["date_death"] = pd.to_datetime(
        df["DATA_MORTE/ULTIMA"], format="%d/%m/%Y", errors="coerce")

    status = df["Morto"].str.strip().str.lower() if "Morto" in df.columns else None
    if status is not None:
        df[cfg.EVENT_COL] = np.where(
            status == "si", 1,
            np.where(status == "no", 0,
                     np.where(df["date_death"].notna(), 1, 0)))
    else:
        df[cfg.EVENT_COL] = df["date_death"].notna().astype(int)

    df.loc[df[cfg.ID_COL].isin(cfg.LOST_TO_FOLLOWUP_IDS), cfg.EVENT_COL] = 0

    undated = (df[cfg.EVENT_COL] == 1) & df["date_death"].isna()
    if undated.any():
        ids = df.loc[undated, cfg.ID_COL].tolist()
        print(f"  WARNING: excluded {len(ids)} patient(s) recorded as dead "
              f"with no date of death: {ids}")
        df = df[~undated].copy()

    lock = pd.Timestamp(cfg.DATA_LOCK)
    end = df["date_death"].fillna(lock)
    df[cfg.DURATION_COL] = (end - df["date_diagnosis"]).dt.days / 30.4375

    negative = df[cfg.DURATION_COL] <= 0
    if negative.any() and "DATA_RMN_BASALE" in df.columns:
        baseline = pd.to_datetime(df["DATA_RMN_BASALE"], format="%d/%m/%Y",
                                  errors="coerce")
        df.loc[negative, cfg.DURATION_COL] = (
            (end[negative] - baseline[negative]).dt.days / 30.4375)
        print(f"  [fix]     {int(negative.sum())} patient(s) timed from "
              f"baseline MRI instead of histological diagnosis")
    df = df[df[cfg.DURATION_COL] > 0]

    df["Eta"] = _to_number(df["Età"])
    ki67_col = "Ki67_%" if "Ki67_%" in df.columns else "Ki67"
    df["Ki67"] = _to_number(df[ki67_col])
    df["MGMT_METILAZIONE"] = _to_number(df["MGMT_METILAZIONE"])
    if cfg.MGMT_BINARY_THRESHOLD is not None:
        df["MGMT_METILAZIONE"] = (
            df["MGMT_METILAZIONE"] >= cfg.MGMT_BINARY_THRESHOLD
        ).astype(float).where(df["MGMT_METILAZIONE"].notna())
        print(f"  MGMT dichotomised at >= {cfg.MGMT_BINARY_THRESHOLD}% "
              f"to match the external centre")
    resection = df["RESEZIONE"].fillna("").str.strip().str.upper()
    df["RESEZIONE"] = (resection == "GTR").astype(int)
    df["GENERE"] = df["GENERE"].str.strip().map({"M": 0, "F": 1})

    p53_col = ("p53_MUTAZIONE" if "p53_MUTAZIONE" in df.columns
               else "p53_ESPRESSIONE")
    p53 = df[p53_col].str.strip().str.lower()
    df["p53_category"] = p53
    df["p53_ESPRESSIONE"] = (~p53.isin(cfg.P53_ABERRANT_CATEGORIES)).astype(float)
    df.loc[p53.isin(("na", "")), "p53_ESPRESSIONE"] = np.nan

    df["IDH"] = df["IDH_MUTAZIONE"].str.strip().str.lower()

    lobe = df["Lobo"].fillna("").str.strip().str.upper()
    df["DEEP"] = lobe.str.contains(cfg.DEEP_PATTERN, regex=True).astype(float)
    n_blank = int((lobe == "").sum())
    if n_blank:
        print(f"  {n_blank} patient(s) with no location recorded, "
              f"treated as non-deep")
    df.loc[lobe == "", "DEEP"] = 0.0

    for pid, value in cfg.TARGET_FIXES.items():
        mask = df[cfg.ID_COL] == pid
        if mask.any() and cfg.TARGET_COL in df.columns:
            old = df.loc[mask, cfg.TARGET_COL].iloc[0]
            df.loc[mask, cfg.TARGET_COL] = value
            print(f"  [fix]     patient {pid}: target {old} -> {value}")

    if cfg.TARGET_COL in df.columns:
        df[cfg.TARGET_COL] = pd.to_numeric(df[cfg.TARGET_COL], errors="coerce")

    if cfg.EXCLUDE_IDS:
        before = len(df)
        df = df[~df[cfg.ID_COL].isin(cfg.EXCLUDE_IDS)]
        print(f"  [exclude] {before - len(df)} patient(s): {cfg.EXCLUDE_IDS}")

    if cfg.REQUIRE_IDH_WILDTYPE:
        before = len(df)
        df = df[df["IDH"] == "wt"]
        if before != len(df):
            print(f"  [exclude] {before - len(df)} IDH-mutant patient(s)")

    return df.reset_index(drop=True)


def load_radiomic() -> pd.DataFrame:
    """Quibim radiomic table, one row per patient."""
    df = pd.read_csv(cfg.DATA_RADIOMIC, encoding="utf-8-sig")
    df[cfg.ID_COL] = pd.to_numeric(df[cfg.ID_COL])

    features = [c for c in df.columns if c != cfg.ID_COL]
    n_missing = int(df[features].isna().sum().sum())
    assert n_missing == 0, (
        f"{n_missing} missing radiomic values. Check that every patient has all "
        f"three regions segmented."
    )

    for prefix in cfg.REGIONS:
        n = sum(c.startswith(f"{prefix}_") for c in df.columns)
        assert n > 0, f"No columns with prefix {prefix!r} found"

    return df


def add_volume_ratios(radiomic: pd.DataFrame) -> pd.DataFrame:
    """Volume ratios between compartments."""
    df = radiomic.copy()
    column = {p: f"{p}_{cfg.VOLUME_FEATURE}" for p in cfg.REGIONS}

    missing = [c for c in column.values() if c not in df.columns]
    if missing:
        print(f"  volume ratios skipped: missing {missing}")
        return df

    enh = df[column["enh"]].to_numpy(dtype=float)
    nec = df[column["nec"]].to_numpy(dtype=float)
    ede = df[column["ede"]].to_numpy(dtype=float)
    eps = 1e-6

    df["ratio_necrosis_enhancing"] = nec / (enh + eps)
    df["ratio_edema_enhancing"] = ede / (enh + eps)
    df["ratio_necrosis_core"] = nec / (nec + enh + eps)
    df["volume_total"] = enh + nec + ede

    print("  added 4 volume-ratio features")
    return df


def field_strength(ids: pd.Series) -> np.ndarray:
    """1.5T or 3T, from the chronological patient numbering."""
    return np.where(ids <= cfg.FIELD_ID_CUTOFF, 1.5, 3.0)


def scanner_affected(values: pd.DataFrame, field: np.ndarray) -> list[str]:
    """Features whose distribution differs between the two scanners."""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    low, high = field == 1.5, field == 3.0
    p = [mannwhitneyu(values.loc[low, c], values.loc[high, c]).pvalue
         for c in values.columns]
    rejected = multipletests(p, method="fdr_bh", alpha=cfg.SCANNER_FDR)[0]
    return [c for c, r in zip(values.columns, rejected) if r]


def combat_adjust(values: pd.DataFrame, field: np.ndarray) -> pd.DataFrame:
    """Location-scale harmonisation by scanner."""
    out = values.copy()
    for batch in np.unique(field):
        mask = field == batch
        if mask.sum() < 5:
            continue
        mean, sd = out.loc[mask].mean(), out.loc[mask].std(ddof=0)
        sd = sd.replace(0, 1.0)
        out.loc[mask] = ((out.loc[mask] - mean) / sd) * values.std(ddof=0) \
            + values.mean()
    return out


class SurvivalDataset:
    """Container for the merged cohort and its feature spaces."""

    RATIO_COLUMNS = [
        "ratio_necrosis_enhancing",
        "ratio_edema_enhancing",
        "ratio_necrosis_core",
        "volume_total",
    ]

    def __init__(self, clinical: pd.DataFrame, radiomic: pd.DataFrame):
        merged = clinical.merge(radiomic, on=cfg.ID_COL, how="inner")

        incomplete = merged[list(cfg.CLINICAL_FEATURES)].isna().any(axis=1)
        if incomplete.any():
            ids = merged.loc[incomplete, cfg.ID_COL].tolist()
            print(f"  excluded {len(ids)} patient(s) with missing clinical "
                  f"values: {ids}")
            merged = merged[~incomplete]

        merged = merged.sort_values(cfg.ID_COL).reset_index(drop=True)

        radiomic_columns = [c for c in merged.columns
                            if any(c.startswith(f"{p}_") for p in cfg.REGIONS)]
        self.scanner_dropped: list[str] = []
        if cfg.SCANNER_CORRECTION != "none" and radiomic_columns:
            field = field_strength(merged[cfg.ID_COL])
            if cfg.SCANNER_CORRECTION == "drop":
                self.scanner_dropped = scanner_affected(
                    merged[radiomic_columns], field)
                print(f"  scanner: dropped {len(self.scanner_dropped)} of "
                      f"{len(radiomic_columns)} features differing between "
                      f"1.5T and 3T")
            elif cfg.SCANNER_CORRECTION == "combat":
                merged[radiomic_columns] = combat_adjust(
                    merged[radiomic_columns], field)
                remaining = scanner_affected(merged[radiomic_columns], field)
                print(f"  scanner: batch-adjusted; {len(remaining)} of "
                      f"{len(radiomic_columns)} features still differ")

        self.df = merged
        self.duration = merged[cfg.DURATION_COL].to_numpy(dtype=float)
        self.event = merged[cfg.EVENT_COL].to_numpy(dtype=int)

        self.clinical_features = list(cfg.CLINICAL_FEATURES)
        self.region_features = {
            prefix: [c for c in radiomic.columns if c.startswith(f"{prefix}_")]
            for prefix in cfg.REGIONS
        }

        example = self.region_features["enh"]
        n_shape = sum(cfg.feature_class(c) == "shape" for c in example)
        if n_shape != len(cfg.SHAPE_FEATURES):
            print(f"  WARNING: {n_shape} shape features matched, expected "
                  f"{len(cfg.SHAPE_FEATURES)}. Check SHAPE_FEATURES in "
                  f"config_cox.py against the export labels.")
        self.ratio_features = [
            c for c in self.RATIO_COLUMNS if c in merged.columns
        ]

        self._validate()

    def _validate(self) -> None:
        missing = self.df[self.clinical_features].isna().sum()
        assert missing.sum() == 0, (
            f"Missing clinical values:\n{missing[missing > 0]}"
        )
        assert (self.duration > 0).all(), "Non-positive survival times present"
        assert self.event.sum() >= 20, (
            f"Only {self.event.sum()} events; a Cox model is not informative "
            f"at this size"
        )

    def columns_for(self, space: str) -> list[str]:
        """Column names making up one feature space."""
        spec = {**cfg.FEATURE_SPACES, **cfg.SECONDARY_SPACES}[space]
        classes = spec.get("classes")

        columns: list[str] = []
        if spec["clinical"]:
            columns += self.clinical_features
        for prefix in spec["regions"]:
            block = self.region_features[prefix]
            if classes is not None:
                block = [c for c in block if cfg.feature_class(c) in classes]
            block = [c for c in block if c not in self.scanner_dropped]
            columns += block
        if len(spec["regions"]) > 1:
            columns += self.ratio_features
        return columns

    def X(self, space: str) -> np.ndarray:
        return self.df[self.columns_for(space)].to_numpy(dtype=float)

    def binary_indices(self, space: str) -> list[int]:
        """Positions of binary clinical variables, for the scaler to skip."""
        columns = self.columns_for(space)
        return [i for i, c in enumerate(columns) if c in cfg.BINARY_CLINICAL]

    @property
    def y_surv(self):
        """Structured array in the format scikit-survival expects."""
        from sksurv.util import Surv

        return Surv.from_arrays(
            event=self.event.astype(bool), time=self.duration)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    def epv(self, space: str) -> float:
        return self.n_events / max(1, len(self.columns_for(space)))


def load_dataset() -> SurvivalDataset:
    """Load, merge, validate and report."""
    print("Loading clinical data...")
    clinical = load_clinical()
    print("Loading radiomic data...")
    radiomic = load_radiomic()

    if cfg.ADD_VOLUME_RATIOS:
        radiomic = add_volume_ratios(radiomic)

    clinical_ids = set(clinical[cfg.ID_COL])
    radiomic_ids = set(radiomic[cfg.ID_COL])
    only_radiomic = sorted(radiomic_ids - clinical_ids)
    only_clinical = sorted(clinical_ids - radiomic_ids)

    if only_radiomic:
        print(f"  WARNING: in radiomics but not clinical: {only_radiomic}")
    if only_clinical:
        print(f"  {len(only_clinical)} patient(s) without radiomics "
              f"(no FLAIR available): {only_clinical}")

    ds = SurvivalDataset(clinical, radiomic)

    print(f"\nCohort: n = {len(ds.df)}, {ds.n_events} deaths, "
          f"{len(ds.df) - ds.n_events} censored")
    print(f"Median OS (deaths): "
          f"{np.median(ds.duration[ds.event == 1]):.1f} months")
    print("\nFeature spaces:")
    for space in cfg.FEATURE_SPACES:
        n = len(ds.columns_for(space))
        print(f"  {space:10s} {n:4d} features   EPV {ds.epv(space):6.2f}")
    return ds


def cohort_summary(ds: SurvivalDataset) -> pd.DataFrame:
    """Descriptive table for the Methods section."""
    from lifelines import KaplanMeierFitter

    df = ds.df
    kmf = KaplanMeierFitter().fit(ds.duration, 1 - ds.event)

    rows = [
        ("Patients", f"{len(df)}", "IDH-wildtype glioblastoma, WHO 2021"),
        ("Deaths", f"{ds.n_events}", ""),
        ("Censored", f"{len(df) - ds.n_events}", "alive at data lock"),
        ("Median follow-up", f"{kmf.median_survival_time_:.1f} months",
         "reverse Kaplan-Meier"),
        ("Median OS (deaths)",
         f"{np.median(ds.duration[ds.event == 1]):.2f} months", ""),
        ("Accrual",
         f"{df['date_diagnosis'].min().date()} to "
         f"{df['date_diagnosis'].max().date()}", ""),
        ("Data lock", cfg.DATA_LOCK, ""),
    ]

    for space in cfg.FEATURE_SPACES:
        rows.append((f"EPV, {space}", f"{ds.epv(space):.2f}",
                     f"{len(ds.columns_for(space))} candidate features"))

    return pd.DataFrame(rows, columns=["Item", "Value", "Note"])
