"""Figures of the main and supplementary results."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config_cox as cfg

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": cfg.FIG_DPI, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

PALETTE = {
    "clinical": "#DD8452",
    "radiomic_shape": "#4C72B0", "mixed_shape": "#55A868",
    "radiomic_shape_fo": "#64B5CD", "radiomic_all": "#4C72B0",
    "mixed_all": "#55A868",
    "mixed_enhancing": "#2E8B57",
    "enhancing_only": "#8172B3", "necrosis_only": "#937860",
    "edema_only": "#DA8BC3",
}


SPACE_ORDER = ["clinical", "radiomic_shape", "mixed_shape", "radiomic_all",
               "radiomic_shape_fo", "enhancing_only", "necrosis_only",
               "edema_only", "mixed_all", "mixed_enhancing"]


def _order(spaces) -> list[str]:
    """Spaces in the order of Table S4, then anything else."""
    listed = [s for s in SPACE_ORDER if s in spaces]
    return listed + [s for s in spaces if s not in listed]


RATIO_LABEL = {
    "ratio_necrosis_enhancing": "Necrosis / enhancing volume",
    "ratio_edema_enhancing": "Edema / enhancing volume",
    "ratio_necrosis_core": "Necrotic fraction of core",
    "volume_total": "Total lesion volume",
}


def _pretty(name: str) -> str:
    """Label for a figure axis."""
    if name in cfg.PRETTY_CLINICAL:
        if name == "MGMT_METILAZIONE" and cfg.MGMT_BINARY_THRESHOLD is not None:
            return f"MGMT methylation (\u2265 {cfg.MGMT_BINARY_THRESHOLD:.0f}%)"
        return cfg.PRETTY_CLINICAL[name]
    return RATIO_LABEL.get(name, name)


def _save(fig, name: str):
    path = cfg.FIGURES_DIR / f"{name}.{cfg.FIG_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path.name}")
    return path


def _label(space: str) -> str:
    return cfg.SPACE_LABELS.get(space, space)


def _fmt_p(p: float, resolution: float | None = None) -> str:
    """p values with three decimals; 'p < 0.001' below that."""
    if resolution is not None and p <= resolution + 1e-12:
        return f"p \u2264 {resolution:.3f}"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_cindex(summary: pd.DataFrame, optimism: pd.DataFrame | None = None,
                spaces=None, name: str = "fig_cox_cindex", title: str | None = None,
                show_title: bool = True):
    """Three estimates of discrimination for each model, side by side."""
    available = list(summary["space"].values)
    order = _order([s for s in (spaces or available) if s in available])
    if not order:
        return None

    def lookup(table, space, column):
        if table is None or table.empty or column not in table.columns:
            return np.nan
        hit = table.loc[table["space"] == space, column]
        return float(hit.iloc[0]) if len(hit) else np.nan

    cv = np.array([lookup(summary, s, "c_test_mean") for s in order])
    sd = np.array([lookup(summary, s, "c_test_sd") for s in order])
    apparent = np.array([lookup(optimism, s, "c_apparent") for s in order])
    corrected = np.array([lookup(optimism, s, "c_corrected") for s in order])
    colours = [PALETTE.get(s, "#777777") for s in order]

    many = len(order) > 4
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(7.5, 1.55 * len(order) + 3), 5.6))
    x = np.arange(len(order))

    series = (("Apparent", apparent, 0.28, -width, None),
              ("Cross-validated", cv, 0.62, 0.0, sd),
              ("Optimism-corrected", corrected, 1.0, width, None))
    for label, values, alpha, offset, err in series:
        ax.bar(x + offset, values, width, color=colours, alpha=alpha,
               edgecolor=colours, linewidth=0.8,
               yerr=err, capsize=3 if err is not None else 0,
               error_kw={"elinewidth": 1, "ecolor": "#555555"})
        for xi, v, e in zip(x + offset, values,
                            err if err is not None else [0] * len(values)):
            if np.isnan(v):
                continue
            ax.text(xi, v + (e if err is not None else 0) + 0.008,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.8 if many else 8.5, rotation=90 if many else 0,
                    fontweight="bold" if label != "Apparent" else "normal",
                    color="#333333")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#555555", alpha=a, label=l)
               for l, a in (("Apparent", 0.28),
                            ("Cross-validated (mean \u00b1 SD)", 0.62),
                            ("Optimism-corrected", 1.0))]
    handles.append(Line2D([0], [0], color="gray", linestyle="--",
                          label="No discrimination (0.5)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8.5,
              ncol=4 if many else 1, title="Colour: model; shade: estimate",
              title_fontsize=8.5, alignment="left")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(s).replace(" (", "\n(") for s in order],
                       fontsize=8.5 if many else 10)
    top = np.nanmax(np.concatenate([apparent, cv + np.nan_to_num(sd)]))
    ax.set_ylim(0.45, top + (0.13 if many else 0.09))
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("Harrell's C-index")
    if show_title:
        n_folds = int(summary["n_folds"].max())
        ax.set_title(title or ("Discrimination by feature space\n"
                               f"Penalized Cox; {n_folds} folds of repeated nested "
                               "cross-validation; 200 bootstrap replicates for optimism"),
                     fontsize=11)
    return _save(fig, name)


def plot_cindex_distribution(folds: pd.DataFrame):
    """Fold-level concordance, showing the spread behind each mean."""
    ok = folds[folds["status"] == "ok"]
    order = _order(list(ok["space"].unique()))

    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 3, 4.8))
    data = [ok.loc[ok["space"] == s, "c_test"].to_numpy() for s in order]
    parts = ax.boxplot(data, patch_artist=True,
                       tick_labels=[_label(s).replace(" (", "\n(") for s in order],
                       medianprops={"color": "black"})
    for patch, space in zip(parts["boxes"], order):
        patch.set_facecolor(PALETTE.get(space, "#777777"))
        patch.set_alpha(0.65)

    rng = np.random.default_rng(cfg.SEED)
    for i, values in enumerate(data, start=1):
        ax.scatter(i + rng.normal(0, 0.05, len(values)), values, s=12,
                   color="black", alpha=0.3)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylabel("Harrell's C-index")
    ax.set_title(f"C-index in each outer test fold "
                 f"({ok.groupby('space').size().max()} folds per model)")
    return _save(fig, "fig_cox_cindex_distribution")


def plot_selection_frequency(frequency: pd.DataFrame, space=None, top_n=30):
    """How often each feature is retained across folds."""
    if space is None:
        space = next((s for s in cfg.FEATURE_SPACES if s.startswith("mixed")),
                     "clinical")
    block = frequency[frequency["space"] == space].head(top_n)
    if block.empty:
        print("  no selection data; skipping")
        return None

    clinical = set(cfg.CLINICAL_FEATURES)
    colours, labels = [], []
    for _, row in block.iterrows():
        name = row["feature"]
        if name in clinical:
            colours.append("#8172B3")
            labels.append(_pretty(name) + "  (clinical)")
        else:
            f = row["frequency"]
            colours.append("#55A868" if f >= 0.8
                           else "#DD8452" if f >= 0.5 else "#B0B0B0")
            labels.append(_pretty(name))

    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.32 * len(block))))
    ax.barh(range(len(block)), block["frequency"] * 100, color=colours)
    ax.set_yticks(range(len(block)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(80, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Selection frequency (% of folds)")
    ax.set_title(
        f"Selection frequency across folds \u2014 {_label(space)} model\n"
        f"{int(block['n_folds'].iloc[0])} outer folds; elastic-net Cox")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#8172B3", label="Clinical variable"),
                       Patch(color="#55A868", label="Imaging, selected in \u2265 80%"),
                       Patch(color="#DD8452", label="Imaging, 50\u201379%"),
                       Patch(color="#B0B0B0", label="Imaging, < 50%")],
              loc="lower right", fontsize=8.5)
    return _save(fig, f"fig_cox_selection_{space}")


def plot_risk_stratification(strat: dict, ds):
    """Kaplan-Meier by risk group of the final mixed model (apparent)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    lines = []

    for label, info in strat["curves"].items():
        info["fitter"].plot_survival_function(
            ax=ax, color=info["colour"], linewidth=2, ci_show=True)
        lines.append(f"{label}: n = {info['n']}, {info['events']} events, "
                     f"median {info['median']:.1f} mo")

    ax.axvline(12, color="gray", linestyle=":", linewidth=1)
    ax.text(12.6, 0.97, "12 months", fontsize=8, color="gray")
    ax.set_xlabel("Time since histological diagnosis (months)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, min(60, float(ds.duration.max())))
    ax.set_title(
        "Overall survival by risk score of the final mixed model\n"
        f"Median split, apparent (same patients used for fitting); "
        f"log-rank {_fmt_p(strat['logrank_p'])}")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.03, 0.12, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                  "alpha": 0.88, "edgecolor": "lightgray"})
    return _save(fig, "fig_cox_risk_stratification")


def plot_rad_score_forest(table: pd.DataFrame, top_n=20):
    """Hazard ratios of the features composing the radiomic score."""
    block = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(block))))

    for i, (_, row) in enumerate(block.iterrows()):
        colour = "#C44E52" if row["coefficient"] > 0 else "#4C72B0"
        ax.plot([1.0, row["hazard_ratio"]], [i, i], color=colour, linewidth=2)
        ax.plot(row["hazard_ratio"], i, "o", color=colour, markersize=7)

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(block)))
    ax.set_yticklabels(
        [_pretty(f) for f in block["feature"]], fontsize=8)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
    lo = min(block["hazard_ratio"].min(), 1.0)
    hi = max(block["hazard_ratio"].max(), 1.0)
    ticks = [t for t in (0.25, 0.33, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25,
                         1.5, 2.0, 3.0, 4.0) if lo / 1.08 <= t <= hi * 1.08]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(lo / 1.1, hi * 1.1)
    ax.set_xlabel("exp(β), penalized coefficients (log scale)")
    ax.set_title(
        "Coefficients of the final mixed model\n"
        "Red increases hazard, blue decreases it")
    return _save(fig, "fig_cox_rad_score_forest")


def plot_permutation(results: dict):
    """Observed concordance against the permutation null."""
    spaces = _order(list(results))
    fig, axes = plt.subplots(1, len(spaces), figsize=(4.8 * len(spaces), 4.2))
    axes = np.atleast_1d(axes)

    for ax, space in zip(axes, spaces):
        res = results[space]
        ax.hist(res["null"], bins=30, color="#8DA0CB", edgecolor="white",
                alpha=0.9, label="Permuted outcomes")
        ax.axvline(res["observed"], color="#C44E52", linewidth=2.5,
                   label=f"Observed C = {res['observed']:.3f}\n"
                         f"{_fmt_p(res['p_value'], res.get('resolution'))}")
        ax.set_xlabel("C-index")
        ax.set_title(_label(space))
        ax.legend(fontsize=8, loc="upper left")
        if ax is axes[0]:
            ax.set_ylabel("Frequency")

    first = results[spaces[0]]
    fig.suptitle(
        f"Permutation test ({first['n']} permutations; "
        f"smallest attainable p = {first['resolution']:.3f})", y=1.03)
    fig.tight_layout()
    return _save(fig, "fig_cox_permutation")


def plot_calibration(tables: dict[float, pd.DataFrame]):
    """Predicted versus observed survival at each horizon."""
    horizons = sorted(tables)
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.6 * len(horizons), 4.6))
    axes = np.atleast_1d(axes)

    for ax, horizon in zip(axes, horizons):
        table = tables[horizon]
        if table["predicted"].isna().all():
            ax.text(0.5, 0.5, "baseline hazard\nunavailable", ha="center",
                    va="center", transform=ax.transAxes, color="gray")
            ax.set_title(f"{horizon:.0f} months")
            continue

        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1,
                label="Perfect calibration")
        ax.errorbar(table["predicted"], table["observed"],
                    yerr=1.96 * table["se"], fmt="o-", color="#4C72B0",
                    markersize=7, capsize=4, label="Model")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted survival")
        ax.set_title(f"{horizon:.0f} months")
        if ax is axes[0]:
            ax.set_ylabel("Observed survival (Kaplan-Meier)")
            ax.legend(loc="upper left", fontsize=8)

    sizes = sorted({int(n) for t in tables.values() for n in t["n"]})
    groups = len(next(iter(tables.values())))
    size_text = (f"{sizes[0]} patients each" if len(sizes) == 1
                 else f"{sizes[0]}\u2013{sizes[-1]} patients each")
    fig.suptitle("A  Apparent calibration of the mixed model\n"
                 f"{groups} groups of predicted risk, {size_text}; "
                 "bars are 95% CIs", y=1.04)
    fig.tight_layout()
    return _save(fig, "fig_cox_calibration")


def plot_orthogonality(table: pd.DataFrame):
    """Which clinical variables the radiomic score does not already encode."""
    if table.empty:
        return None

    block = table.sort_values("p", ascending=False)
    labels = [_pretty(v) for v in block["variable"]]
    logp = -np.log10(block["p"].clip(lower=1e-12))
    colours = ["#C44E52" if p < 0.05 else "#4C72B0" for p in block["p"]]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(block))))
    ax.barh(range(len(block)), logp, color=colours, alpha=0.9)
    ax.axvline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1)
    ax.text(-np.log10(0.05) + 0.05, -0.6, "p = 0.05", fontsize=8, color="gray")

    ax.set_yticks(range(len(block)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$-\log_{10}(p)$ for association with the radiomic score")
    ax.set_title(
        "Association of the out-of-fold radiomic score with "
        "clinical-molecular variables")

    for i, (p, e) in enumerate(zip(block["p"], block["effect"])):
        ax.text(logp.iloc[i] + 0.05, i, _fmt_p(p), va="center",
                fontsize=8, color="#555555")
    ax.set_xlim(0, max(logp.max() * 1.35, 2))
    return _save(fig, "fig_cox_orthogonality")


def plot_time_dependent_auc(table: pd.DataFrame):
    """Discrimination at fixed horizons."""
    if table.empty:
        return None

    spaces = [sp for sp in cfg.FEATURE_SPACES if sp in set(table["space"])]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k, space in enumerate(spaces):
        block = table[table["space"] == space].sort_values("horizon_months")
        dx = (k - (len(spaces) - 1) / 2) * 0.35
        ax.errorbar(block["horizon_months"] + dx, block["auc_mean"],
                    yerr=block["auc_sd"], marker="o", capsize=4, linewidth=2,
                    color=PALETTE.get(space, "#777777"), label=_label(space))

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(sorted(table["horizon_months"].unique()))
    ax.set_xlabel("Horizon (months from diagnosis)")
    ax.set_ylabel("Cumulative/dynamic AUC")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Time-dependent discrimination\n"
                 "Mean across outer folds; error bars show between-fold SD")
    ax.legend(loc="upper left", ncol=len(spaces), fontsize=9)

    at_risk = table.drop_duplicates("horizon_months").set_index(
        "horizon_months")["at_risk"]
    for h, n in at_risk.items():
        ax.annotate(f"{n} at risk", (h, 0.32), ha="center", fontsize=8,
                    color="gray")
    return _save(fig, "fig_cox_time_dependent_auc")


def plot_decision_curve(table: pd.DataFrame):
    """Net benefit against the two default strategies."""
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colours = {"Clinical": PALETTE["clinical"],
               "Radiomic score": PALETTE.get("radiomic_shape", "#4C72B0"),
               "Combined": PALETTE.get("mixed_shape", "#55A868")}
    names = {"Combined": "Clinical + radiomic score"}

    for name, colour in colours.items():
        if name in table.columns:
            ax.plot(table["threshold"], table[name], linewidth=2,
                    color=colour, label=names.get(name, name))

    ax.plot(table["threshold"], table["treat_all"], "--", color="gray",
            linewidth=1.2, label="Treat all")
    ax.plot(table["threshold"], table["treat_none"], "-", color="black",
            linewidth=1, label="Treat none")

    lowest = min(0.0, min(table[c].min() for c in colours if c in table))
    ax.set_ylim(lowest - 0.05, max(table["treat_all"].max(), 0.4) + 0.05)
    ax.set_xlabel("Threshold probability of death by "
                  f"{table.attrs.get('horizon', 12):.0f} months")
    ax.set_ylabel("Net benefit")
    prevalence = table.attrs.get("prevalence")
    if prevalence is None:
        t, nb = float(table["threshold"].iloc[0]), float(table["treat_all"].iloc[0])
        prevalence = (nb + t / (1 - t)) / (1 + t / (1 - t))
    ax.set_title("B  Decision curve analysis (out-of-fold predictions)\n"
                 f"Kaplan–Meier mortality by {table.attrs.get('horizon', 12):.0f} "
                 f"months: {100 * prevalence:.0f}%")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, "fig_cox_decision_curve")


def plot_proportional_hazards(table: pd.DataFrame):
    """Schoenfeld test per covariate; bars past the line break the assumption."""
    if table.empty:
        return None

    block = table.sort_values("p", ascending=False)
    logp = -np.log10(block["p"].clip(lower=1e-12))
    colours = ["#C44E52" if p < 0.05 else "#4C72B0" for p in block["p"]]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(block))))
    ax.barh(range(len(block)), logp, color=colours, alpha=0.9)
    ax.axvline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(block)))
    ax.set_yticklabels([_pretty(v) for v in block["variable"]], fontsize=9)
    ax.set_xlabel(r"$-\log_{10}(p)$, scaled Schoenfeld residuals")
    ax.set_title("Proportional hazards assumption\n"
                 "Red: hazard ratio not constant over follow-up")
    for i, p in enumerate(block["p"]):
        ax.text(logp.iloc[i] + 0.05, i, _fmt_p(p), va="center",
                fontsize=8, color="#555555")
    ax.set_xlim(0, max(logp.max() * 1.35, 2))
    return _save(fig, "fig_cox_proportional_hazards")


def plot_time_varying(table: pd.DataFrame, panel: str = ""):
    """Hazard ratio of one covariate before and after the split point."""
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (_, row) in enumerate(table.iterrows()):
        colour = "#C44E52" if row["p"] < 0.05 else "#9A9A9A"
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i], color=colour,
                linewidth=2.2)
        ax.plot(row["hazard_ratio"], i, "o", color=colour, markersize=9)
        ptxt = "p < 0.001" if row["p"] < 0.001 else f"p = {row['p']:.3f}"
        ax.text(1.02, i, f"HR {row['hazard_ratio']:.2f} "
                         f"({row['ci_lower']:.2f}\u2013{row['ci_upper']:.2f}), "
                         f"{ptxt}, {row['events']} events",
                transform=ax.get_yaxis_transform(), va="center", fontsize=11,
                color=colour)

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["window"], fontsize=11)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelsize=11)
    ax.set_ylim(len(table) - 0.4, -0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Hazard ratio, methylated vs unmethylated (log scale)",
                  fontsize=11)
    if panel:
        ax.set_title(panel, loc="left", fontsize=14, fontweight="bold")
        if "p_interaction" in table.columns:
            ax.text(1.0, 1.02, "p for interaction = "
                    f"{float(table['p_interaction'].iloc[0]):.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
    else:
        ax.set_title(f"{_pretty(table['variable'].iloc[0])}: effect by follow-up window",
                     fontsize=11)
    return _save(fig, "fig_cox_time_varying")


def plot_landmark(table: pd.DataFrame, panel: str = ""):
    """Discrimination by window of follow-up: early (truncated) to late (landmark)."""
    if table.empty:
        return None
    windows = (table.drop_duplicates("population").sort_values("order"))
    populations = windows["population"].tolist()
    spaces = _order(list(table["space"].unique()))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(populations))
    series = {}
    for space in spaces:
        block = table[table["space"] == space].set_index("population")
        values = [block.loc[p, "c_index"] if p in block.index else np.nan
                  for p in populations]
        series[space] = values
        ax.plot(x, values, marker="o", markersize=8, linewidth=2.4,
                color=PALETTE.get(space, "#777777"), label=_label(space))

    w, h = 0.21, 0.012
    path_points = []
    for values in series.values():
        for j in range(len(x) - 1):
            if np.isnan(values[j]) or np.isnan(values[j + 1]):
                continue
            t = np.linspace(0, 1, 60)
            path_points += list(zip(x[j] + t, values[j] + t * (values[j + 1] - values[j])))
    path_points = np.array(path_points) if path_points else np.empty((0, 2))
    markers = [(xi, v) for values in series.values() for xi, v in zip(x, values)
               if not np.isnan(v)]
    placed = []

    def hits(box):
        x0, x1, y0, y1 = box
        m = 0.004
        inside = ((path_points[:, 0] > x0 - 0.02) & (path_points[:, 0] < x1 + 0.02)
                  & (path_points[:, 1] > y0 - m) & (path_points[:, 1] < y1 + m)).sum()
        dots = sum((x0 - 0.07 < mx < x1 + 0.07) and (y0 - 0.011 < my < y1 + 0.011)
                   for mx, my in markers)
        labels = sum(not (x1 < b[0] or x0 > b[1] or y1 < b[2] or y0 > b[3])
                     for b in placed)
        return inside + 50 * dots + 50 * labels

    for j, xi in enumerate(x):
        ranked = sorted(((series[s][j], s) for s in spaces
                         if not np.isnan(series[s][j])), reverse=True)
        for v, space in ranked:
            candidates = []
            for dy in (0.016, -0.016, 0.026, -0.026):
                yc = v + dy
                candidates.append((xi - w / 2, xi + w / 2, yc - h / 2, yc + h / 2))
            for dx in (0.09, -0.09 - w):
                for dy in (0.0, 0.012, -0.012):
                    yc = v + dy
                    candidates.append((xi + dx, xi + dx + w, yc - h / 2, yc + h / 2))
            best = min(candidates, key=hits)
            placed.append(best)
            ax.text((best[0] + best[1]) / 2, (best[2] + best[3]) / 2, f"{v:.3f}",
                    ha="center", va="center", fontsize=8,
                    color=PALETTE.get(space, "#777777"), fontweight="bold")

    early = windows["window"].eq("early").sum()
    ax.axvspan(-0.5, early - 0.5, color="#C44E52", alpha=0.06)
    ax.axvspan(early + 0.5, len(populations) - 0.5, color="#4C72B0", alpha=0.06)
    ax.text((early - 1) / 2, 0.415, "early mortality", ha="center",
            fontsize=9, color="#C44E52", style="italic")
    ax.text((early + len(populations)) / 2, 0.415, "survivors",
            ha="center", fontsize=9, color="#4C72B0", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.population}\n({r.events} deaths)"
                        for r in windows.itertuples()], fontsize=8.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0.4, max(0.75, table["c_index"].max() + 0.05))
    ax.set_xlim(-0.5, len(populations) - 0.5)
    ax.set_ylabel("Harrell's C-index (out-of-fold)")
    if panel:
        ax.set_title(panel, loc="left", fontsize=14, fontweight="bold")
    else:
        ax.set_title("Discrimination by window of follow-up\n"
                     "Early: follow-up truncated at the horizon; "
                     "survivors: patients alive at the horizon")
    ax.legend(loc="upper center", ncol=len(spaces), fontsize=9,
              bbox_to_anchor=(0.5, -0.16))
    return _save(fig, "fig_cox_landmark")


def plot_crossover_lines(windows: pd.DataFrame, crossover: pd.DataFrame):
    """Interaction plot of the C-index in the early window and among survivors."""
    if windows.empty:
        return None

    horizons = sorted(windows["horizon_months"].unique())
    main = [s for s in ("radiomic_shape", "clinical") if s in set(windows.space)]
    extra = [s for s in windows["space"].unique()
             if s not in main and s.startswith("mixed")]

    fig, axes = plt.subplots(1, len(horizons), figsize=(7.4 * len(horizons), 6.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    offsets = {s: d for s, d in zip(main + extra, (-0.035, 0.035, 0.0))}

    for ax, horizon in zip(axes, horizons):
        block = windows[windows["horizon_months"] == horizon]
        x = np.array([0.0, 1.0])

        for space in extra:
            b = block[block.space == space].set_index("window").loc[["early", "late"]]
            ax.plot(x + offsets[space], b["c_index"], color="#9A9A9A",
                    linestyle="--", linewidth=1.6, marker="o", markersize=6,
                    zorder=2, label=_label(space))

        for space in main:
            b = block[block.space == space].set_index("window").loc[["early", "late"]]
            colour = PALETTE.get(space, "#777777")
            xs = x + offsets[space]
            ax.errorbar(xs, b["c_index"],
                        yerr=[b["c_index"] - b["ci_lower"],
                              b["ci_upper"] - b["c_index"]],
                        fmt="none", ecolor=colour, elinewidth=1.3, capsize=4,
                        alpha=0.55, zorder=3)
            ax.plot(xs, b["c_index"], color=colour, linewidth=3.2, marker="o",
                    markersize=11, zorder=4, label=_label(space))
            for xi, v, side in zip(xs, b["c_index"], (-1, 1)):
                ax.text(xi + side * 0.09, v, f"{v:.3f}", ha="right" if side < 0
                        else "left", va="center", fontsize=14,
                        fontweight="bold", color=colour, zorder=5)

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, zorder=1)
        ax.text(0.5, 0.505, "chance", ha="center", va="bottom", fontsize=11,
                color="gray")

        info = block.drop_duplicates("window").set_index("window")
        ax.set_xticks(x)
        ax.set_xticklabels([
            f"Deaths within\n{horizon:.0f} months\n({info.loc['early', 'deaths']} deaths)",
            f"Survivors beyond\n{horizon:.0f} months\n({info.loc['late', 'n']} patients, "
            f"{info.loc['late', 'deaths']} deaths)"], fontsize=13)
        ax.set_xlim(-0.45, 1.45)

        row = crossover[crossover["horizon_months"] == horizon]
        if len(row):
            r = row.iloc[0]
            significant = r["bootstrap_p"] < r["bonferroni_alpha"]
            p_text = _fmt_p(r["bootstrap_p"])
            ax.set_title(f"{horizon:.0f}-month landmark\n"
                         + ("Crossover significant, " if significant
                            else "Same direction, not significant, ")
                         + p_text, fontsize=15,
                         color="black" if significant else "#555555")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=13)

    axes[0].set_ylabel("C-index (out-of-fold)", fontsize=14)
    axes[0].set_ylim(0.35, 0.92)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(_label(s)) for s in main + extra if _label(s) in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=len(order), frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=13.5)
    fig.suptitle("Discrimination in the early window and among survivors",
                 fontsize=17, y=1.03)
    fig.text(0.5, -0.13,
             "Bars: bootstrap 95% CI (2000 resamples). p-value: test of the "
             "crossover, i.e. whether the radiomic-minus-clinical difference "
             "changes between the two windows. Exploratory analysis.",
             ha="center", fontsize=11, color="#555555")
    fig.tight_layout()
    return _save(fig, "fig_cox_crossover_lines")


def plot_crossover(table: pd.DataFrame):
    """Relative performance of the radiomic and clinical models across follow-up."""
    if table.empty:
        return None

    horizons = table["horizon_months"].tolist()
    colours = ["#2B4C7E", "#8A8A8A", "#B08D57"][:len(horizons)]
    has_ci = "early_ci_lower" in table.columns
    alpha_bonf = float(table["bonferroni_alpha"].iloc[0])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.6),
                                  gridspec_kw={"width_ratios": [1.6, 1]})

    positions, labels = [], []
    lows, highs = [0.0], [0.0]
    for k, (row, colour) in enumerate(zip(table.itertuples(), colours)):
        x = np.array([k * 3.0, k * 3.0 + 1.35])
        y = np.array([row.advantage_early, row.advantage_survivors])
        ax.plot(x, y, color=colour, linewidth=2.2, zorder=2)
        if has_ci:
            lo = y - np.array([row.early_ci_lower, row.survivors_ci_lower])
            hi = np.array([row.early_ci_upper, row.survivors_ci_upper]) - y
            ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=colour,
                        elinewidth=1.4, capsize=5, alpha=0.75, zorder=1)
            lows += [row.early_ci_lower, row.survivors_ci_lower]
            highs += [row.early_ci_upper, row.survivors_ci_upper]
        else:
            lows += list(y)
            highs += list(y)
        ax.plot(x, y, "o", color=colour, markersize=10, zorder=3)
        for xi, yi in zip(x, y):
            ax.text(xi + 0.09, yi, f"{yi:+.3f}", va="center", fontsize=12,
                    fontweight="bold", color=colour)
        positions += list(x)
        labels += [f"Early window\n(deaths within {row.horizon_months:.0f} mo,"
                   f"\n{row.deaths_early_window} deaths)",
                   f"Survivors\n(alive at {row.horizon_months:.0f} mo,"
                   f"\n{row.deaths_survivors} deaths)"]

    span = max(highs) - min(lows)
    bottom, top = min(lows) - 0.08 * span, max(highs) + 0.08 * span
    ax.axhspan(0, top, color=PALETTE["radiomic_shape"], alpha=0.07, zorder=0)
    ax.axhspan(bottom, 0, color=PALETTE["clinical"], alpha=0.07, zorder=0)
    ax.axhline(0, color="black", linewidth=1.2, zorder=1)
    ax.text(0.99, 0.97, "radiomic model better", transform=ax.transAxes,
            va="top", ha="right", fontsize=13, style="italic", color=PALETTE["radiomic_shape"])
    ax.text(0.01, 0.03, "clinical model better", transform=ax.transAxes,
            va="bottom", fontsize=13, style="italic", color=PALETTE["clinical"])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(bottom, top)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(-0.6, positions[-1] + 0.9)
    ax.set_ylabel("C-index: radiomic minus clinical", fontsize=13)
    ax.set_title("Advantage of the radiomic model, by window of follow-up",
                 fontsize=14)

    ys = np.arange(len(horizons))[::-1]
    for y, row, colour in zip(ys, table.itertuples(), colours):
        ax2.plot([row.ci_lower, row.ci_upper], [y, y], color=colour,
                 linewidth=2.6)
        ax2.plot(row.crossover, y, "o", color=colour, markersize=11)
        significant = row.bootstrap_p < alpha_bonf
        ax2.text(1.03, y + 0.11, f"{row.crossover:+.3f}", transform=
                 ax2.get_yaxis_transform(), va="center", fontsize=12.5,
                 fontweight="bold", color=colour)
        ax2.text(1.03, y - 0.19, _fmt_p(row.bootstrap_p)
                 + ("  *" if significant else ""),
                 transform=ax2.get_yaxis_transform(), va="center",
                 fontsize=12, color=colour)
    ax2.axvline(0, color="black", linewidth=1.2)
    ax2.tick_params(axis="x", labelsize=13)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"{h:.0f} months" for h in horizons], fontsize=13)
    ax2.set_ylim(-0.7, len(horizons) - 0.3)
    ax2.set_xlabel("Crossover\n(early advantage minus survivor advantage)", fontsize=13)
    ax2.set_title("Did the advantage reverse?", fontsize=14)
    ax2.text(0.5, -0.34, f"* below the Bonferroni threshold "
             f"(\u03b1 = {alpha_bonf:.3f})", transform=ax2.transAxes,
             ha="center", fontsize=10.5, color="#555555")

    fig.suptitle("Relative performance of the radiomic and clinical models "
                 "across follow-up (out-of-fold predictions)", fontsize=16,
                 y=1.03)
    fig.text(0.5, -0.09,
             "Bars: bootstrap 95% CI (2000 resamples of patients). Exploratory "
             "analysis; the variability of model fitting is not propagated.",
             ha="center", fontsize=10.5, color="#555555")
    fig.tight_layout()
    return _save(fig, "fig_cox_crossover")


def plot_pair_types(table: pd.DataFrame, name="fig_cox_pair_types",
                    title="Concordance by type of patient pair"):
    """C-index restricted to early-early, early-late and late-late pairs."""
    if table.empty:
        return None
    labels = {"early_vs_early": "Early vs early", "early_vs_late": "Early vs late",
              "late_vs_late": "Late vs late"}
    kinds = [k for k in labels if k in set(table["pair_type"])]
    spaces = _order(list(table["space"].unique()))
    width = 0.8 / len(spaces)

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, space in enumerate(spaces):
        block = table[table["space"] == space].set_index("pair_type")
        values = [block.loc[p, "c_index"] for p in kinds]
        x = np.arange(len(kinds)) + (k - (len(spaces) - 1) / 2) * width
        ax.bar(x, values, width, color=PALETTE.get(space, "#777777"),
               label=_label(space), alpha=0.9)
    pairs = table.drop_duplicates("pair_type").set_index("pair_type")["pairs"]
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels([f"{labels[p]}\n({pairs[p]} pairs)" for p in kinds],
                       fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0.3, 0.9)
    ax.set_ylabel("C-index within pair type")
    ax.set_title(f"{title}\nEarly = death within 12 months")
    ax.legend(loc="upper right")
    return _save(fig, name)


def plot_sample_size(table: pd.DataFrame):
    """Width of the 95% interval of the C-index against validation size."""
    if table.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for space in _order(list(table["space"].unique())):
        block = table[table["space"] == space]
        ax.plot(block["n"], block["ci_width"], marker="o", linewidth=2,
                color=PALETTE.get(space, "#777777"), label=_label(space))
    ax.axhline(0.10, color="gray", linestyle="--", linewidth=1)
    ax.text(table["n"].max(), 0.105, "\u00b10.05", fontsize=8, color="gray",
            ha="right")
    ax.set_xlabel("Patients in the validation cohort")
    ax.set_ylabel("Width of the 95% interval of the C-index")
    ax.set_title("Precision of an external validation by cohort size\n"
                 "Resampled from the derivation case mix")
    ax.legend(loc="upper right")
    return _save(fig, "fig_cox_sample_size")


def plot_montecarlo(results: pd.DataFrame):
    """Distribution of paired differences across random splits."""
    if results.empty:
        return None
    mixed = next((s for s in cfg.FEATURE_SPACES if s.startswith("mixed")), None)
    others = [s for s in cfg.FEATURE_SPACES if s != mixed and s in results]
    fig, axes = plt.subplots(1, len(others), figsize=(5 * len(others), 4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, other in zip(axes, others):
        diff = (results[mixed] - results[other]).dropna()
        ax.hist(diff, bins=25, color=PALETTE.get(other, "#777777"), alpha=0.75,
                edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(diff.median(), color="#C44E52", linewidth=2)
        ax.set_title(f"{_label(mixed)} minus {_label(other)}\n"
                     f"better in {(diff > 0).mean() * 100:.0f}% of splits; "
                     f"median {diff.median():+.3f}", fontsize=10)
        ax.set_xlabel("Difference in C-index")
    axes[0].set_ylabel("Number of splits")
    fig.suptitle(f"Monte Carlo cross-validation ({len(results)} paired "
                 f"random 80/20 splits)", y=1.03)
    fig.tight_layout()
    return _save(fig, "fig_cox_montecarlo")


def plot_shap(shap_result: dict):
    """Beeswarm of Shapley values for the final model."""
    try:
        import shap
    except ImportError:
        print("  shap not installed; skipping")
        return None

    values = shap_result["values"]
    if values.shape[1] == 0:
        print("  no non-zero coefficients; skipping SHAP figure")
        return None

    height = max(4.5, 0.42 * values.shape[1])
    fig = plt.figure(figsize=(9, height))
    shap.summary_plot(values, shap_result["X"],
                      feature_names=shap_result["names"], show=False)
    plt.title("SHAP values — mixed penalized Cox model\n"
              "Positive values increase predicted hazard", fontsize=10)
    plt.xlabel("SHAP value")
    plt.tight_layout()
    return _save(fig, "fig_cox_shap")
