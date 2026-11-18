"""Figure 4: forest plot of the external validation of the three frozen models."""
import config_cox as cfg

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SPACES = ["clinical", "radiomic_shape", "mixed_shape"]
COLOURS = {"clinical": "#DD8452", "radiomic_shape": "#4C72B0",
           "mixed_shape": "#55A868"}


def main() -> int:
    folder = cfg.RESULTS_DIR / "external_validation"
    results = {}
    for space in SPACES:
        path = folder / f"validation_{space}_common.json"
        if not path.exists():
            sys.exit(f"{path} not found: run validate_external.py first.")
        results[space] = json.loads(path.read_text(encoding="utf-8"))

    derivation = {}
    cv_path = cfg.RESULTS_DIR / "cox_cv_summary.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        derivation = dict(zip(cv["space"], cv["c_test_mean"]))

    labels = [cfg.SPACE_LABELS.get(s, s) for s in SPACES]
    y = list(range(len(SPACES)))[::-1]

    fig, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(11, 4.0),
                                     gridspec_kw={"width_ratios": [1.15, 1]})

    for yi, space in zip(y, SPACES):
        r, colour = results[space], COLOURS[space]
        lo, hi = r["c_index_ci"]
        ax_c.plot([lo, hi], [yi, yi], color=colour, linewidth=2.4)
        ax_c.plot(r["c_index"], yi, "o", color=colour, markersize=10, zorder=3)
        if r.get("c_expected") is not None:
            ax_c.plot(r["c_expected"], yi, "D", markerfacecolor="white",
                      markeredgecolor=colour, markersize=8, zorder=4)
        if space in derivation:
            ax_c.plot(derivation[space], yi, "X", color="#555555",
                      markersize=8, zorder=4)
        ax_c.text(1.02, yi, f"{r['c_index']:.3f} ({lo:.3f}\u2013{hi:.3f})",
                  va="center", fontsize=9, color=colour,
                  transform=ax_c.get_yaxis_transform())

        hlo, hhi = r["hr_per_sd_ci"]
        ax_h.plot([hlo, hhi], [yi, yi], color=colour, linewidth=2.4)
        ax_h.plot(r["hr_per_sd"], yi, "s", color=colour, markersize=9, zorder=3)
        p = r["hr_per_sd_p"]
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax_h.text(1.02, yi, f"{r['hr_per_sd']:.2f} ({hlo:.2f}\u2013{hhi:.2f}), {ptxt}",
                  va="center", fontsize=9, color=colour,
                  transform=ax_h.get_yaxis_transform())

    ax_c.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax_c.axvline(0.60, color="gray", linestyle="--", linewidth=1)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels)
    ax_c.set_xlim(0.35, 0.90)
    ax_c.set_xlabel("Harrell's C-index (95% CI)")
    ax_c.set_title("A  Discrimination in the external cohort", loc="left",
                   fontsize=10)
    handles = [
        plt.Line2D([], [], marker="o", color="black", linestyle="", label="External"),
        plt.Line2D([], [], marker="D", markerfacecolor="white", color="black",
                   linestyle="", label="Expected from case mix"),
        plt.Line2D([], [], marker="X", color="#555555", markersize=8,
                   linestyle="", label="Derivation (CV)"),
    ]
    ax_c.legend(handles=handles, fontsize=8, loc="upper center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, -0.28))

    ax_h.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_h.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax_h.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
    ax_h.xaxis.set_major_formatter(ScalarFormatter())
    ax_h.xaxis.set_minor_locator(NullLocator())
    ax_h.set_yticks(y)
    ax_h.set_yticklabels([])
    ax_h.set_xlabel("Hazard ratio per SD of the frozen risk score (95% CI)")
    ax_h.set_title("B  Association with survival", loc="left", fontsize=10)

    for ax in (ax_c, ax_h):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = cfg.FIGURES_DIR / "fig_validation_forest.png"
    fig.savefig(out, dpi=cfg.FIG_DPI, bbox_inches="tight")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
