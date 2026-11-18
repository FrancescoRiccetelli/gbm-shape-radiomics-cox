"""Figure 1: study flowchart with counts read from the data files."""
from __future__ import annotations

import config_cox as cfg

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def derivation_counts() -> list[tuple[str, int, str]]:
    """Counts at each step of the derivation cohort."""
    from data_cox import (_read_clinical_csv, _read_ods, load_clinical,
                          load_dataset, load_radiomic)

    path = cfg.DATA_CLINICAL
    raw = (_read_ods(path) if str(path).lower().endswith(".ods")
           else _read_clinical_csv(path))
    clinical = load_clinical()
    radiomic_ids = set(load_radiomic()[cfg.ID_COL])
    with_radiomics = clinical[cfg.ID_COL].isin(radiomic_ids).sum()
    ds = load_dataset()

    return [
        ("Patients in the clinical database", len(raw), ""),
        ("Eligible after data checks", len(clinical),
         f"excluded: {len(raw) - len(clinical)} (inconsistent or incomplete "
         f"survival data, IDH-mutant)"),
        ("With multiregional segmentation", int(with_radiomics),
         f"excluded: {len(clinical) - int(with_radiomics)} (FLAIR unavailable "
         f"or not segmentable)"),
        ("Analyzed", len(ds.df),
         f"excluded: {int(with_radiomics) - len(ds.df)} (missing clinical "
         f"variable)\n{ds.n_events} deaths, {len(ds.df) - ds.n_events} censored"),
    ]


def external_counts(clinical_path, radiomic_path) -> list[tuple[str, int, str]]:
    """Counts at each step of the external cohort."""
    from freeze_model import MODEL_PATH
    import validate_external as ve

    frozen = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    raw = (pd.read_excel(clinical_path) if str(clinical_path).endswith(".xlsx")
           else pd.read_csv(clinical_path))
    frame = ve.load_external(clinical_path, radiomic_path, cfg.DATA_LOCK, frozen)

    clinical_cols = list(frozen["clinical_features"])
    complete = frame[clinical_cols].notna().all(axis=1)
    events_complete = int(frame.loc[complete, cfg.EVENT_COL].sum())

    censored_complete = int(complete.sum()) - events_complete
    return [
        ("Patients provided", len(raw), ""),
        ("With multiregional segmentation", len(frame),
         f"excluded: {len(raw) - len(frame)} (FLAIR unavailable)"),
        ("Analyzed", int(complete.sum()),
         f"excluded: {len(frame) - int(complete.sum())} (incomplete clinical "
         f"or molecular data)\n{events_complete} deaths, {censored_complete} "
         f"censored"),
    ]


def _compact(steps):
    """Drop intermediate steps that excluded nobody; they add only clutter."""
    kept = [steps[0]]
    for i, (label, n, note) in enumerate(steps[1:], start=1):
        last = i == len(steps) - 1
        if n != kept[-1][1]:
            kept.append((label, n, note))
        elif last:
            previous_exclusion = [l for l in kept[-1][2].split("\n")
                                  if l.startswith("excluded")]
            extra = [l for l in note.split("\n")
                     if l and not l.startswith("excluded")]
            kept[-1] = (label, n, "\n".join(previous_exclusion + extra))
        elif note and "\n" in note:
            kept[-1] = (kept[-1][0], kept[-1][1], note.split("\n", 1)[1])
    return kept


def draw_column(ax, x, steps, title, colour, n_rows):
    """One vertical flow of boxes, with exclusions noted beneath each arrow."""
    ax.text(x, 0.98, title, ha="center", va="top", fontsize=12,
            fontweight="bold", color=colour)
    ys = np.linspace(0.86, 0.10, n_rows)
    for k, (label, n, note) in enumerate(steps):
        y = ys[k]
        lines = note.split("\n") if note else []
        exclusion = next((l for l in lines if l.startswith("excluded")), None)
        extra = [l for l in lines if not l.startswith("excluded")]
        box = f"{label}\nn = {n}" + ("".join(f"\n{l}" for l in extra))
        ax.text(x, y, box, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white",
                      "edgecolor": colour, "linewidth": 1.5})
        if k > 0:
            ax.annotate("", xy=(x, y + 0.06), xytext=(x, ys[k - 1] - 0.06),
                        arrowprops={"arrowstyle": "->", "color": "gray"})
            if exclusion:
                ax.text(x + 0.02, (y + ys[k - 1]) / 2, exclusion, ha="left",
                        va="center", fontsize=8, color="#8B0000",
                        style="italic")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-clinical", default=None)
    parser.add_argument("--external-radiomic", default=None)
    args = parser.parse_args()

    columns = [("Derivation cohort", _compact(derivation_counts()), "#4C72B0")]
    if args.external_clinical and args.external_radiomic:
        columns.append(("External validation cohort",
                        _compact(external_counts(args.external_clinical,
                                                 args.external_radiomic)),
                        "#DD8452"))

    n_rows = max(len(steps) for _, steps, _ in columns)
    fig, ax = plt.subplots(figsize=(6.5 * len(columns), 1.6 * n_rows + 1.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for i, (title, steps, colour) in enumerate(columns):
        draw_column(ax, (i + 0.5) / len(columns) - 0.08, steps, title, colour,
                    n_rows)

    path = cfg.FIGURES_DIR / "fig_cox_flowchart.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path.name}")
    for title, steps, _ in columns:
        print(f"\n{title}")
        for label, n, note in steps:
            print(f"  {label}: {n}" + (f"   [{note.splitlines()[0]}]" if note else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
