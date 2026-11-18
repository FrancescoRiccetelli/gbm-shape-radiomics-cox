"""Redraws the main figures from the saved result tables."""
import config_cox as cfg

import subprocess
import sys

import pandas as pd


def main() -> int:
    import figures_cox as fg

    needed = ["cox_cv_summary.csv", "cox_optimism.csv", "cox_landmark.csv",
              "cox_time_varying_effect.csv"]
    missing = [n for n in needed if not (cfg.RESULTS_DIR / n).exists()]
    if missing:
        sys.exit(f"Missing in results_cox: {missing}. Run the pipeline first.")

    read = lambda n: pd.read_csv(cfg.RESULTS_DIR / n, encoding="utf-8-sig")
    fg.plot_cindex(read("cox_cv_summary.csv"), read("cox_optimism.csv"),
                   spaces=list(cfg.FEATURE_SPACES), name="fig_cox_cindex",
                   show_title=False)
    fg.plot_landmark(read("cox_landmark.csv"), panel="A")
    fg.plot_time_varying(read("cox_time_varying_effect.csv"), panel="B")

    forest = cfg.BASE_DIR / "figure_external_forest.py"
    subprocess.run([sys.executable, str(forest)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
