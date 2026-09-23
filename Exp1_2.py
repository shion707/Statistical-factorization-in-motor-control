"""Behavioural results for Experiments 1 and 2: three prior widths x two response
contingencies.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import common


# condition -> (response mode, prior range in degrees, half range in degrees)
CONDITIONS = {
    "d180": ("Delayed", 180, 90), "d120": ("Delayed", 120, 60), "d60": ("Delayed", 60, 30),
    "f180": ("Free", 180, 90), "f120": ("Free", 120, 60), "f60": ("Free", 60, 30),
}
# Order of the per-participant polynomial that removes systematic bias before the
# variability and serial-dependence analyses.
POLY_ORDER = {"d180": 15, "d120": 15, "d60": 15, "f180": 15, "f120": 15, "f60": 15}
RANGE_NAME = {180: "Wide", 120: "Medium", 60: "Narrow"}
SIMPLE_HALF = {"s120": 60, "s60": 30}
BIN_CENTRES = np.arange(-9, 10) * 20  # 20-degree ΔTarget bins, -180 to +180


# ----------------------------------------------------------------- local helpers
def group_mean_sem(values):
    """Mean and SEM down the participant axis of a participant-by-bin matrix.
    """
    values = np.asarray(values, float)
    n_participants = values.shape[0]
    mean = np.full(values.shape[1], np.nan)
    error = np.full(values.shape[1], np.nan)
    for column in range(values.shape[1]):
        finite = values[np.isfinite(values[:, column]), column]
        if finite.size:
            mean[column] = finite.mean()
        if finite.size > 1:
            error[column] = finite.std(ddof=1) / np.sqrt(n_participants)
    return mean, error


def smooth_bins(values):
    values = np.asarray(values, float)
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        window = values[max(0, i - 1):i + 2]
        window = window[np.isfinite(window)]
        if window.size:
            out[i] = window.mean()
    return out


# ----------------------------------------------------------------- analysis
def summarize(reaching, condition):
    """Bias curve, bias slope, motor variability and reaction time for one condition."""
    response, prior, half = CONDITIONS[condition]
    subset = reaching.loc[reaching["condition"] == condition]
    targets, participants, error = common.target_means(subset, "error_deg")
    _, _, rt = common.target_means(subset, "rt_ms")
    n = len(participants)

    error_smoothed = common.moving_mean(error)
    bias = -np.nanmean(error_smoothed, axis=1)
    bias = bias - np.nanmean(bias)
    # Slope of the raw per-target means, pooled over participants; negated so that a
    # positive slope means attraction towards the centre of the prior.  Its error bar is
    # the between-participant SEM of the individual slopes, as for panels 3f and 3g.
    slope, _ = common.linear_slope(np.repeat(targets, n), error.ravel())
    slope_se = common.sem(common.slope_per_column(targets, error))

    # Motor variability: residual scatter about the per-participant polynomial fit of
    # trial-wise error on target.
    target_matrix, _ = common.matrix(subset, "target_index")
    error_matrix, _ = common.matrix(subset, "error_deg")
    residual = common.detrend(target_matrix, error_matrix, POLY_ORDER[condition])
    participant_sd = np.sqrt(np.nanmean(residual ** 2, axis=0))

    rt_smoothed = common.moving_mean(rt)
    participant_rt = np.nanmean(rt_smoothed, axis=0)
    return dict(
        condition=condition, response=response, prior=prior, half=half, n=n,
        target=targets - half, bias=bias, bias_se=common.sem(error_smoothed, axis=1),
        slope=-slope, slope_se=slope_se,
        sd=np.nanmean(participant_sd), sd_se=common.sem(participant_sd),
        rt_mean=np.nanmean(rt_smoothed, axis=1), rt_se=common.sem(rt_smoothed, axis=1),
        rt_avg=np.nanmean(participant_rt), rt_avg_se=common.sem(participant_rt),
    )


def simple_rt(data, condition):
    """Target profile of simple reaction time, for the dashed baselines of panel 3d."""
    targets, _, rt = common.target_means(data.loc[data["condition"] == condition], "rt_ms")
    rt = common.moving_mean(rt)
    return targets - SIMPLE_HALF[condition], np.nanmean(rt, axis=1), common.sem(rt, axis=1)


def serial_dependence(reaching, condition):
    """One-back deviation as a function of ΔTarget, per participant, in 20-degree bins.
    """
    response, prior, half = CONDITIONS[condition]
    subset = reaching.loc[reaching["condition"] == condition]
    target, participants = common.matrix(subset, "target_index")
    error, _ = common.matrix(subset, "error_deg")

    residual = common.detrend(target, error, POLY_ORDER[condition])
    residual = common.remove_outliers_2d(residual, 3)

    # ΔTarget is previous minus current, wrapped onto (-180, 180], then binned.
    delta = common.wrap180(target[:-1] - target[1:])
    bin_index = common.round_half_away(delta / 20)

    subject = np.full((len(participants), len(BIN_CENTRES)), np.nan)
    for column, centre in enumerate(BIN_CENTRES / 20):
        for row in range(len(participants)):
            values = residual[1:, row][bin_index[:, row] == centre]
            values = common.remove_outliers(values, 2.5)
            values = values[np.isfinite(values)]
            if values.size:
                subject[row, column] = values.mean()
    return dict(condition=condition, response=response, prior=prior, half=half,
                n=len(participants), x=BIN_CENTRES, subject=subject)


def dependence_index(result):
    """deviation at negative ΔTarget minus deviation at positive ΔTarget."""
    subject = result["subject"] - np.nanmean(result["subject"])
    if result["prior"] == 60:
        index = subject[:, 8] - subject[:, 10]  # narrow range: only +/-20 degrees is populated
    else:
        index = (np.nanmean(subject[:, [7, 8]], axis=1)
                 - np.nanmean(subject[:, [10, 11]], axis=1))
    # Every participant is kept: outliers are already trimmed trial-by-trial within each
    # bin above, and a second trim at the participant level would drop subjects from the
    # index while the curves in 5g still show them.
    return index[np.isfinite(index)]


# ----------------------------------------------------------------- figures
def figure1(bias, serial):
    """Panels 1e and 1g for the wide-prior, delayed-response cohort."""
    colour = common.COLOUR[180]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), constrained_layout=True)
    fig.suptitle("Wide prior (180°), delayed response", fontsize=10, y=1.03)

    ax = axes[0]
    target, curve, error = bias["target"], bias["bias"], bias["bias_se"]
    # The fitted line is the pooled linear bias, drawn about its own mean.
    fit = bias["slope"] * (target - target.mean())
    ax.fill_between(target, curve - error, curve + error, color=colour, alpha=.20, lw=0)
    ax.plot(target, curve, color=colour, lw=2)
    ax.plot(target, fit, color=colour, lw=1.6)
    ax.axhline(0, color=".35", ls=(0, (9, 6)), lw=.8)
    ax.axvline(0, color=".35", ls=(0, (9, 6)), lw=.8)
    for x, y, word in [(-84, 5.6, "Repulsive"), (28, 5.6, "Attractive"),
                       (-84, -6.5, "Attractive"), (32, -6.5, "Repulsive")]:
        ax.text(x, y, word, color=".75", fontweight="bold")
    ax.set(xlim=(-95, 95), ylim=(-7, 7), xticks=[-80, -40, 0, 40, 80], yticks=[-6, -3, 0, 3, 6],
           xlabel="Target (°)", ylabel="Systematic bias (°)", title="Attractive bias")
    common.label(ax, "e")

    ax = axes[1]
    # Panel 1g omits the -180° bin, so it is centred on the remaining 18 bins.
    curve, error = group_mean_sem(serial["subject"][:, 1:] - np.nanmean(serial["subject"][:, 1:]))
    delta = serial["x"][1:]
    visible = np.abs(delta) <= 100
    ax.fill_between(delta[visible], (curve - error)[visible], (curve + error)[visible],
                    color=colour, alpha=.20, lw=0)
    ax.plot(delta[visible], curve[visible], color=colour, lw=2)
    ax.axhline(0, color=".35", ls=(0, (9, 6)), lw=.8)
    ax.axvline(0, color=".35", ls=(0, (9, 6)), lw=.8)
    for x, y, word in [(-95, 1.22, "Repulsive"), (20, 1.22, "Attractive"),
                       (-95, -1.45, "Attractive"), (24, -1.45, "Repulsive")]:
        ax.text(x, y, word, color=".75", fontweight="bold")
    ax.set(xlim=(-110, 110), ylim=(-1.55, 1.55), xticks=[-90, 0, 90], yticks=[-1, 0, 1],
           xlabel="ΔTarget (°)", ylabel="Deviation (°)", title="Repulsive sequential effect")
    common.label(ax, "g")
    return fig, curve


def figure3(results, simple):
    """Panels 3b-g."""
    fig = plt.figure(figsize=(11, 7.6))
    grid = fig.add_gridspec(2, 3, hspace=.42, wspace=.36, left=.06, right=.985, top=.90, bottom=.09)

    for column, (letter, title, keys) in enumerate([
        ("b", "Exp 2a: Delay Response", ["d180", "d120", "d60"]),
        ("c", "Exp 2b: Free Response", ["f180", "f120", "f60"]),
    ]):
        ax = fig.add_subplot(grid[0, column])
        common.label(ax, letter)
        for key in keys:
            result = results[key]
            colour = common.COLOUR[result["prior"]]
            ax.fill_between(result["target"], result["bias"] - result["bias_se"],
                            result["bias"] + result["bias_se"], color=colour, alpha=.22, lw=0)
            ax.plot(result["target"], result["bias"], color=colour, lw=2)
        ax.axhline(0, color="k", lw=.6, ls="--")
        ax.axvline(0, color="k", lw=.6, ls="--")
        ax.set(xlim=(-95, 95), ylim=(-7, 7), xticks=[-80, -40, 0, 40, 80],
               yticks=[-6, -3, 0, 3, 6], xlabel="Target (°)", ylabel="Bias (°)", title=title)

    ax = fig.add_subplot(grid[0, 2])
    common.label(ax, "d")
    for key in ["f180", "f120", "f60"]:
        result = results[key]
        colour = common.COLOUR[result["prior"]]
        ax.fill_between(result["target"], result["rt_mean"] - result["rt_se"],
                        result["rt_mean"] + result["rt_se"], color=colour, alpha=.20, lw=0)
        ax.plot(result["target"], result["rt_mean"], color=colour, lw=2)
    for key, prior in [("s120", 120), ("s60", 60)]:
        target, mean, error = simple[key]
        ax.fill_between(target, mean - error, mean + error, color=common.COLOUR[prior], alpha=.14, lw=0)
        ax.plot(target, mean, color=common.COLOUR[prior], lw=1.6, ls=(0, (3, 2)))
    ax.text(0, 320, "Simple RT", ha="center", fontsize=8, color=".35")
    ax.set(xlim=(-95, 95), ylim=(300, 460), xticks=[-80, -40, 0, 40, 80],
           yticks=[300, 350, 400, 450], xlabel="Target (°)", ylabel="RT (ms)",
           title="Exp 2b: Free Response")

    groups = [(180, "d180", "f180"), (120, "d120", "f120"), (60, "d60", "f60")]
    width = .36
    for letter, key_name, position, axis_label, limits, ticks in [
        ("e", "slope", (1, 0), "Slope", (0, .32), [0, .1, .2, .3]),
        ("f", "sd", (1, 1), "Standard Deviation (°)", (0, 8.5), [0, 2, 4, 6, 8]),
    ]:
        ax = fig.add_subplot(grid[position])
        common.label(ax, letter)
        for i, (prior, delayed, free) in enumerate(groups):
            for x, key, colour in [(i - width / 2, delayed, common.PALE[prior]),
                                   (i + width / 2, free, common.COLOUR[prior])]:
                ax.bar(x, results[key][key_name], width, color=colour, edgecolor="k", lw=.7)
                ax.errorbar(x, results[key][key_name], results[key][key_name + "_se"],
                            color="k", lw=1.1, capsize=0)
        ax.set(xticks=[0, 1, 2], xticklabels=["180", "120", "60"], xlabel="Prior Range (°)",
               ylabel=axis_label, ylim=limits, yticks=ticks)
        if letter == "e":
            ax.legend(handles=[Patch(facecolor="0.75", edgecolor="k", label="Delayed"),
                               Patch(facecolor="0.35", edgecolor="k", label="Free")],
                      frameon=False, fontsize=8, loc="upper left", handlelength=1.2)

    ax = fig.add_subplot(grid[1, 2])
    common.label(ax, "g")
    for i, (prior, _, free) in enumerate(groups):
        ax.bar(i, results[free]["rt_avg"], .6, color=common.COLOUR[prior], edgecolor="k", lw=.7)
        ax.errorbar(i, results[free]["rt_avg"], results[free]["rt_avg_se"], color="k", lw=1.1, capsize=0)
    ax.set(xticks=[0, 1, 2], xticklabels=["180", "120", "60"], xlabel="Prior Range (°)",
           ylabel="RT (ms)", ylim=(300, 460), yticks=[300, 350, 400, 450])

    fig.suptitle("Figure 3 — behavioural data", fontsize=12, y=.975)
    fig.text(.5, .008, "colours: Wide=teal · Medium=orange · Narrow=navy   |   "
                       "e/f bars: pale=Delayed, strong=Free", ha="center", fontsize=7.5, color=".4")
    return fig


def figure5g(serial):
    """Panel 5g: serial-dependence curves for delayed and free responses."""
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 4.1), sharey=True, constrained_layout=True)
    for ax, response, keys in [(axes[0], "Delayed", ["d180", "d120", "d60"]),
                               (axes[1], "Free", ["f180", "f120", "f60"])]:
        for key in keys:
            result = serial[key]
            mean, error = group_mean_sem(result["subject"] - np.nanmean(result["subject"]))
            # The free-response curves are noisier and are drawn three-bin smoothed.
            if response == "Free" and result["prior"] != 60:
                mean = smooth_bins(mean)
            visible = np.abs(result["x"]) <= 100
            colour = common.COLOUR[result["prior"]]
            ax.fill_between(result["x"][visible], (mean - error)[visible], (mean + error)[visible],
                            color=colour, alpha=.18, lw=0)
            ax.plot(result["x"][visible], mean[visible], color=colour, lw=2.2,
                    label=RANGE_NAME[result["prior"]])
        ax.axhline(0, color=".25", lw=.8, ls=(0, (9, 6)))
        ax.axvline(0, color=".25", lw=.8, ls=(0, (9, 6)))
        ax.set(xlim=(-115, 115), ylim=(-1.55, 1.55), xticks=[-90, 0, 90], yticks=[-1, 0, 1],
               xlabel="ΔTarget (°)", title=f"{response} Response")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Deviation (°)")
    fig.suptitle("Figure 5g — repulsive serial dependence", fontsize=12, y=1.03)
    return fig


# ----------------------------------------------------------------- entry point
def main():
    common.style()
    reaching = common.load("exp1_2")
    simple_table = common.load("Simple_rt")
    simple = {key: simple_rt(simple_table, key) for key in SIMPLE_HALF}

    results = {key: summarize(reaching, key) for key in CONDITIONS}
    serial = {key: serial_dependence(reaching, key) for key in CONDITIONS}

    for key, result in results.items():
        print(f"{key}: N={result['n']}, slope={result['slope']:.3f}±{result['slope_se']:.3f}, "
              f"SD={result['sd']:.2f}, RT={result['rt_avg']:.1f} ms")

    fig, one_back = figure1(results["d180"], serial["d180"])
    print(f"Figure 1e/g: N={results['d180']['n']}, "
          f"bias slope={np.polyfit(results['d180']['target'], results['d180']['bias'], 1)[0]:.3f} °/°, "
          f"one-back range={np.nanmax(one_back) - np.nanmin(one_back):.2f}°")
    common.show(fig, "Exp1_2_fig1")

    common.show(figure3(results, simple), "Exp1_2_fig3")

    for key in CONDITIONS:
        index = dependence_index(serial[key])
        print(f"{key}: serial-dependence index {index.mean():.2f}±"
              f"{index.std(ddof=1) / np.sqrt(len(index)):.2f}° (n={len(index)})")
    common.show(figure5g(serial), "Exp1_2_fig5g")


if __name__ == "__main__":
    main()
