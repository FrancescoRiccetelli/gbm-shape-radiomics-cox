"""Runs every stage of the analysis in order and writes a summary report."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import config_cox as cfg

HERE = Path(__file__).resolve().parent
LOGS = cfg.LOGS_DIR
RESULTS = cfg.RESULTS_DIR
FIGURES = cfg.FIGURES_DIR
CKPT = cfg.CKPT_DIR
REPORT = HERE / "REPORT_finale.txt"

DATA_FILES = [cfg.DATA_CLINICAL, cfg.DATA_RADIOMIC,
              cfg.EXTERNAL_CLINICAL, cfg.EXTERNAL_RADIOMIC]
EXTERNAL = ["--clinical", str(cfg.EXTERNAL_CLINICAL), "--radiomic",
            str(cfg.EXTERNAL_RADIOMIC), "--data-lock", "2026-09-10"]

STEPS = [
    (1, "diagnostics", ["run_cox.py", "--stage", "diagnostics"]),
    (2, "cv", ["run_cox.py", "--stage", "cv"]),
    (3, "secondary_cv", ["run_cox.py", "--secondary", "--stage", "cv"]),
    (4, "secondary_supplementary",
     ["run_cox.py", "--secondary", "--stage", "supplementary"]),
    (5, "checks", ["run_cox.py", "--stage", "checks"]),
    (6, "montecarlo", ["run_cox.py", "--stage", "montecarlo"]),
    (7, "permutation", ["run_cox.py", "--stage", "permutation"]),
    (8, "figures", ["run_cox.py", "--stage", "figures"]),
    (9, "freeze", ["freeze_model.py", "freeze"]),
    (10, "external_validation", ["validate_external.py"] + EXTERNAL),
    (11, "flowchart", ["figure_flowchart.py", "--external-clinical",
                       str(cfg.EXTERNAL_CLINICAL), "--external-radiomic",
                       str(cfg.EXTERNAL_RADIOMIC)]),
]

SAFETY_HINTS = {
    "flagged binary but takes values":
        "A binary column is no longer where the scaler expects it. Do not "
        "work around it: it means columns have shifted.",
    "binary index out of range":
        "An earlier step removed columns and the binary positions no longer "
        "match. Do not work around it.",
}

CLINICAL_CV_FLOOR = 0.58


def prevent_sleep() -> None:
    """Keep Windows awake while the pipeline runs."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print("Sleep is blocked for as long as this run lasts.")
    except Exception as exc:
        print(f"Could not block sleep ({exc}); disable it in Windows.")


def _content_lines(path: Path) -> list[str]:
    """Non-empty log lines, without the flood of library warnings."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [l.rstrip() for l in lines
            if l.strip() and "warn" not in l.lower()]


def preflight(force: bool, first_step: int) -> list[str]:
    """Check data files, pipeline version and output folders before a run."""
    problems = []
    for name in DATA_FILES:
        if not Path(name).exists():
            problems.append(f"missing data file: {name}")

    pipelines = HERE / "pipelines_cox.py"
    if pipelines.exists():
        text = pipelines.read_text(encoding="utf-8")
        if "tolerance" not in text or "np.percentile(variances" in text:
            problems.append("pipelines_cox.py is not the corrected version "
                            "(VarianceFilter needs `tolerance` and must not "
                            "use np.percentile(variances)")
    else:
        problems.append("missing pipelines_cox.py")

    if not force and first_step == 1:
        for folder in (CKPT, RESULTS, FIGURES):
            if folder.exists() and any(folder.rglob("*.*")):
                problems.append(f"{folder.name} already contains files: a "
                                f"fresh run would silently reuse them")
    return problems


def run_step(number: int, name: str, argv: list[str], heartbeat: float = 300.0):
    """Run one stage, log its output, and report progress while it runs."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{number:02d}_{name}.log"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1")
    started = time.time()
    last_beat = started

    with open(log, "w", encoding="utf-8") as handle:
        process = subprocess.Popen([sys.executable, *argv], cwd=HERE,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   env=env)
        while process.poll() is None:
            time.sleep(2)
            if time.time() - last_beat >= heartbeat:
                tail = _content_lines(log)
                minutes = (time.time() - started) / 60
                print(f"    ... {minutes:5.1f} min  "
                      f"{tail[-1][:90] if tail else '(no output yet)'}",
                      flush=True)
                last_beat = time.time()

    return process.returncode, log, time.time() - started


def check_cv() -> tuple[bool, str]:
    """Check that the cross-validated clinical C-index is above the expected floor."""
    import pandas as pd

    path = RESULTS / "cox_cv_summary.csv"
    if not path.exists():
        return False, "cox_cv_summary.csv was not written"
    table = pd.read_csv(path).set_index("space")["c_test_mean"]
    summary = ", ".join(f"{k} {v:.3f}" for k, v in table.items())
    clinical = table.get("clinical")
    if clinical is None:
        return False, f"no clinical row in the summary ({summary})"
    if clinical < CLINICAL_CV_FLOOR:
        return False, (f"clinical C-index {clinical:.3f} is below "
                       f"{CLINICAL_CV_FLOOR}: the preprocessing correction is "
                       f"probably not active ({summary})")
    return True, summary


def write_report() -> None:
    """Collect every table that matters into one text file."""
    import pandas as pd

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)

    lines: list[str] = []

    def section(title: str) -> None:
        lines.extend(["", "=" * 78, title, "=" * 78])

    def show(name: str, head: int | None = None, drop=()) -> None:
        path = RESULTS / name
        section(name)
        if not path.exists():
            lines.append("(not produced)")
            return
        table = pd.read_csv(path)
        table = table.drop(columns=[c for c in drop if c in table.columns])
        if head:
            table = table.head(head)
        lines.append(table.round(4).to_string(index=False))

    show("cox_cv_summary.csv")
    show("cox_optimism.csv")
    show("cox_permutation.csv")
    show("cox_space_comparison.csv")
    show("cox_space_comparison_all.csv", head=15)
    show("cox_montecarlo.csv")
    show("cox_crossover.csv", drop=("note",))
    show("cox_crossover_power.csv", drop=("note",))
    show("cox_landmark.csv")
    show("cox_time_interaction.csv")
    show("cox_pair_types.csv")
    show("cox_proportional_hazards.csv")
    show("cox_time_varying_effect.csv")
    show("cox_field_subcohorts.csv")
    show("cox_biopsy_sensitivity.csv")
    show("cox_time_dependent_auc.csv")
    show("cox_orthogonality.csv")
    show("cox_rad_score.csv", head=20)

    for name in ("cox_likelihood_ratio_oof.json", "cox_likelihood_ratio.json"):
        section(name)
        lr = RESULTS / name
        lines.append(json.dumps(json.loads(lr.read_text()), indent=2)
                     if lr.exists() else "(not produced)")

    section("frozen_model.json - coefficients")
    frozen = cfg.MODEL_PATH
    if frozen.exists():
        model = json.loads(frozen.read_text(encoding="utf-8"))
        clinical = model["clinical_features"]
        for space in ("clinical", "mixed_shape", "radiomic_shape"):
            spec = model["spaces"].get(space)
            if not spec:
                continue
            coefficients = spec["coefficients"]
            lines.append(f"\n{space}: {len(coefficients)} non-zero "
                         f"coefficients, penalty l1_ratio={spec['l1_ratio']}, "
                         f"alpha={spec['alpha']:.5f}")
            for name in clinical:
                if space != "radiomic_shape":
                    value = coefficients.get(name)
                    lines.append(f"   {name:18s} "
                                 + (f"{value:+.4f}" if value is not None
                                    else "ZEROED"))
            others = {k: v for k, v in coefficients.items()
                      if k not in clinical}
            if others:
                lines.append(f"   + {len(others)} non-clinical features")
    else:
        lines.append("(not produced)")

    section("External validation - console output")
    log = LOGS / "10_external_validation.log"
    lines.extend(_content_lines(log) if log.exists() else ["(not produced)"])

    section("Files produced")
    for folder in (FIGURES, RESULTS / "external_validation"):
        names = (sorted(p.name for p in folder.glob("*.png"))
                 if folder.exists() else [])
        lines.append(f"\n{folder}  ({len(names)} images)")
        lines.extend(f"   {n}" for n in names)

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--from-step", type=int, default=1)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        write_report()
        return 0

    steps = ([s for s in STEPS if s[0] == args.only] if args.only
             else [s for s in STEPS if s[0] >= args.from_step])
    if not steps:
        print("No such stage.")
        return 2

    print(f"Folder : {HERE}")
    print(f"Python : {sys.executable}")

    problems = preflight(args.force, steps[0][0])
    if problems:
        print("\nCannot start:")
        for p in problems:
            print(f"  - {p}")
        return 2

    if args.dry_run:
        print("\nStages that would run:")
        for number, name, argv in steps:
            print(f"  {number:2d}  {name:24s} {' '.join(argv)}")
        print("\nPreflight passed.")
        return 0

    prevent_sleep()
    started = time.time()
    print(f"\nStarting at {time.strftime('%H:%M:%S')}\n")

    for number, name, argv in steps:
        print(f"[{number:2d}/{STEPS[-1][0]}] {name} ...", flush=True)
        code, log, seconds = run_step(number, name, argv)
        if code != 0:
            print(f"\n  FAILED after {seconds / 60:.1f} min (exit code {code})")
            text = log.read_text(encoding="utf-8", errors="replace")
            for phrase, hint in SAFETY_HINTS.items():
                if phrase in text:
                    print(f"  SAFETY CHECK: {hint}")
            print("  Last lines of the log:")
            for line in _content_lines(log)[-8:]:
                print(f"    {line[:110]}")
            print(f"\n  Full log: {log}")
            print(f"  After fixing the cause, resume with:  "
                  f"run_pipeline.py --from-step {number}")
            return 1

        print(f"      done in {seconds / 60:.1f} min")
        if name == "cv":
            ok, message = check_cv()
            print(f"      C-index (cross-validated): {message}")
            if not ok:
                print(f"\n  STOP: {message}")
                return 1

    write_report()
    complete = steps[0][0] == 1 and steps[-1][0] == STEPS[-1][0]
    elapsed = (time.time() - started) / 3600
    if complete:
        print(f"\nAll {len(steps)} stages finished in {elapsed:.2f} hours.")
        print("Send REPORT_finale.txt for review.")
    else:
        print(f"\nStages {steps[0][0]}-{steps[-1][0]} finished in "
              f"{elapsed:.2f} hours. The report reflects only what exists "
              f"so far; it is complete once all 11 stages have run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
