"""Experiment 3: the empirical panels (c-h) of Figure 6.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t, ttest_ind

from common import (load, round_half_away, matrix, detrend, label, remove_outliers,
                    remove_outliers_2d, sem, show, style, wrap180)


ORDER = ["random", "medium", "high"]
COLOUR = {"random": "#8b8e8b", "medium": "#33a58c", "high": "#e6820f"}
PSWITCH = {"random": "1", "medium": "1/2", "high": "1/3"}
# poly_order: order of the per-participant polynomial that removes target-dependent bias.
# miniblock: measure transitions only from the final trial of each mini-block.
# lookup_offset: the fitted bias is read from a 1-degree table at target + lookup_offset.
SETTINGS = {
    "random": dict(poly_order=14, miniblock=False, lookup_offset=1),
    "medium": dict(poly_order=14, miniblock=True, lookup_offset=1),
    "high": dict(poly_order=14, miniblock=True, lookup_offset=1),
}
BINS = (np.arange(1, 14) - 7) * 30  # 13 delta-target bin centres, -180 to +180


# ----------------------------------------------------------------- Experiment 3
def scaled(degrees):
    """Map the 1-360 degree axis onto [-1, 1], for the polynomial fit below."""
    return (np.asarray(degrees, dtype=float) - 180.5) / 179.5


def systematic_deviation(target, error, poly_order, lookup_offset):
    """Remove each participant's target-dependent bias.
    """
    # A few hundred logged targets fall outside 1-360 (0, and 361-364).  Wrap them
    # onto the circle before fitting.
    target = np.mod(target - 1, 360) + 1
    deviation = np.full_like(error, np.nan)
    grid = np.arange(1, 361)
    for subject in range(error.shape[1]):
        valid = np.isfinite(target[:, subject]) & np.isfinite(error[:, subject])
        if valid.sum() <= poly_order:
            continue
        fit = np.polyfit(scaled(target[valid, subject]), error[valid, subject], poly_order)
        lookup = np.polyval(fit, scaled(grid))
        index = np.mod(np.nan_to_num(target[:, subject], nan=0).astype(int) - 1 + lookup_offset, 360)
        bias = lookup[index]
        bias[~np.isfinite(target[:, subject])] = np.nan
        deviation[:, subject] = error[:, subject] - bias
    return remove_outliers_2d(deviation, 2.5)


def miniblock_reference(target):
    """Keep the target only on the final trial of each repeated mini-block."""
    reference = target.astype(float).copy()
    # Within a mini-block the target moves by a few degrees at most, so consecutive targets
    # less than 6 degrees apart count as the same mini-block.  The first three trials of a
    # session are never used as a reference.
    for subject in range(target.shape[1]):
        repeated = np.abs(wrap180(target[:-1, subject] - target[1:, subject])) < 6
        final = np.zeros(target.shape[0], dtype=bool)
        for trial in range(3, target.shape[0] - 1):
            final[trial] = repeated[trial - 1] and not repeated[trial]
        reference[~final, subject] = np.nan
    return reference


def serial_dependence(deviation, reference, target, n_lags=10):
    """Per-participant serial-dependence functions and indices for lags 1-10."""
    n_trials, n_subjects = deviation.shape
    functions = np.full((n_subjects, len(BINS), n_lags), np.nan)
    indices = np.full((n_subjects, n_lags), np.nan)
    for lag in range(1, n_lags + 1):
        bin_index = round_half_away(wrap180(reference[:n_trials - lag] - target[lag:]) / 30) + 7
        current = deviation[lag:]
        function = np.full((n_subjects, len(BINS)), np.nan)
        for subject in range(n_subjects):
            for number in range(1, len(BINS) + 1):
                values = remove_outliers(current[bin_index[:, subject] == number, subject], 2.5)
                if np.isfinite(values).any():
                    function[subject, number - 1] = np.nanmean(values)
        function -= np.nanmean(function)
        edge = np.nanmean(function[:, [0, -1]], axis=1)   # -180 and +180 are one bin
        function[:, 0] = function[:, -1] = edge
        functions[:, :, lag - 1] = function
        indices[:, lag - 1] = remove_outliers(function[:, 5] - function[:, 7], 2.5)

    return functions, indices


RETENTION_LAGS = 5      # the decay is fitted to lags 1-5; beyond that the index is noise


def retention_rate(group_index, n_lags=RETENTION_LAGS):
    """Fit the geometric decay s(k) = amplitude * retention ** (k - 1).
    """
    curve = np.asarray(group_index, dtype=float)[:n_lags]
    if np.isfinite(curve).sum() < 3:
        return np.nan
    lag = np.arange(len(curve))
    objective = lambda p: np.nansum((curve - p[0] * np.clip(p[1], 0, 1) ** lag) ** 2)
    start = [curve[np.isfinite(curve)][0], 0.5]
    best = minimize(objective, start, method="Nelder-Mead",
                    options=dict(xatol=1e-4, fatol=1e-5, maxiter=400))
    return float(np.clip(best.x[1], 0, 1))


def retention_error(participant_index, n_boot=2000, seed=0):
    """Standard error of the group retention rate, by bootstrap over participants.

    The rate is a property of the group curve -- the individual curves are far too noisy
    beyond the second lag to fit one each -- so the bootstrap resamples participants and
    refits the group curve, rather than averaging per-participant fits.
    """
    rng = np.random.default_rng(seed)
    estimates = np.full(n_boot, np.nan)
    for iteration in range(n_boot):
        sample = participant_index[rng.integers(0, len(participant_index), len(participant_index))]
        estimates[iteration] = retention_rate(np.nanmean(sample, axis=0))
    return np.nanstd(estimates, ddof=1)


def analyse_condition(data, condition):
    rows = data.loc[data["condition"] == condition]
    target, participants = matrix(rows, "target_deg")
    error, _ = matrix(rows, "error_deg")
    setting = SETTINGS[condition]
    deviation = systematic_deviation(target, error, setting["poly_order"], setting["lookup_offset"])
    reference = miniblock_reference(target) if setting["miniblock"] else target
    functions, participant_index = serial_dependence(deviation, reference, target)
    n = len(participants)

    keep = np.isfinite(participant_index[:, 0])
    function_1back = functions[keep, :, 0]
    function_2to5 = np.nanmean(functions[:, :, 1:5], axis=2)
    # Same +/-30 window as the per-lag index above, so panels d, f and h share one measure.
    index_2to5 = remove_outliers(function_2to5[:, 5] - function_2to5[:, 7], 2.5)

    return dict(
        n=n, n_curve=int(keep.sum()),
        group_index=np.nanmean(participant_index, axis=0),
        group_index_sem=sem(participant_index, axis=0),
        retention=retention_rate(np.nanmean(participant_index, axis=0)),
        retention_sem=retention_error(participant_index),
        function_1back=np.nanmean(function_1back, axis=0),
        function_1back_sem=sem(function_1back, axis=0),
        function_2to5=np.nanmean(function_2to5, axis=0),
        function_2to5_sem=sem(function_2to5, axis=0),
        index_2to5=np.nanmean(index_2to5), index_2to5_sem=sem(index_2to5),
    )


# ----------------------------------------------------------------- panel c
def compensation(observed_bias, sequential_effect):
    """Share of the target-dependent bias that the sequential effect offsets, in %."""
    bias = np.nansum(np.abs(observed_bias))
    effect = np.nansum(np.abs(sequential_effect))
    return 100 * effect / (bias + effect) if bias + effect > 0 else np.nan


TEST_BLOCK = 3          # blocks 1-2 of the UDL dataset are baseline, before any repetition


def compensation_udl():
    """Percent compensation in the UDL design (Tsay et al. 2022, their Exp 2).
    """
    data = load("udl_reaching")
    data = data.loc[(data["experiment"] == "E2") & np.isfinite(data["aligned_error_deg"])
                    & (data["aligned_error_deg"].abs() < 90)]
    sizes = np.array([30., 60., 90.])
    values = []
    for participant, subset in data.groupby("participant", sort=True):
        subset = subset.sort_values(["block", "trial"])
        base = subset.loc[subset["block"] < TEST_BLOCK]
        test = subset.loc[subset["block"] >= TEST_BLOCK]

        base_relative = base["relative_target_deg"].to_numpy(float)
        base_error = base["aligned_error_deg"].to_numpy(float)
        relative = test["relative_target_deg"].to_numpy(float)
        error = test["aligned_error_deg"].to_numpy(float)

        # Attractive bias, baseline-subtracted target by target.
        bias = np.array([np.nanmean(error[np.abs(relative) == size])
                         - np.nanmean(base_error[np.abs(base_relative) == size])
                         for size in sizes])
        deviation = error - np.nanmean(error[relative == 0])

        previous = np.r_[np.nan, relative[:-1]]
        consecutive = np.r_[False, (np.diff(test["trial"].to_numpy(float)) == 1)
                            & (np.diff(test["block"].to_numpy(float)) == 0)]
        after_probe = (relative == 0) & (previous != 0) & consecutive
        effect = np.full(len(sizes), np.nan)
        for index, size in enumerate(sizes):
            chosen = after_probe & (np.abs(previous) == size)
            if chosen.sum() >= 2:
                signed = deviation[chosen]
                signed[previous[chosen] < 0] *= -1
                effect[index] = np.nanmean(signed)
        if np.isfinite(bias).all() and np.isfinite(effect).all():
            values.append(compensation(bias, effect))
    return np.array(values)


def compensation_random():
    """The same model-free measure for the Experiment-1 delayed random design."""
    rows = load("exp1_2")
    rows = rows.loc[rows["condition"] == "d180"]  # the wide-prior delayed condition
    target, participants = matrix(rows, "target_deg")
    error, _ = matrix(rows, "error_deg")
    deviation = remove_outliers_2d(detrend(target, error, 20), 3)

    slope = np.full(len(participants), np.nan)
    for subject in range(len(participants)):
        ok = np.isfinite(target[:, subject]) & np.isfinite(error[:, subject])
        means = pd.Series(error[ok, subject]).groupby(target[ok, subject]).mean()
        if len(means) > 10:
            slope[subject] = -np.polyfit(means.index.to_numpy(float), means.to_numpy(float), 1)[0]

    edges = np.arange(-9, 10)
    bin_index = round_half_away(wrap180(target[:-1] - target[1:]) / 20)
    function = np.full((len(participants), len(edges)), np.nan)
    for subject in range(len(participants)):
        for column, number in enumerate(edges):
            values = remove_outliers(deviation[1:, subject][bin_index[:, subject] == number], 2.5)
            if np.isfinite(values).any():
                function[subject, column] = np.nanmean(values)
    function -= np.nanmean(function)

    grid = np.arange(-90, 91, 5.)
    values = []
    for subject in range(len(participants)):
        ok = np.isfinite(function[subject])
        if ok.sum() < 8 or not np.isfinite(slope[subject]):
            continue
        # Expected sequential pull on each target, averaging over possible predecessors.
        expected = np.array([np.mean(np.interp(wrap180(grid - current), edges[ok] * 20,
                                               function[subject, ok], left=0, right=0))
                             for current in grid])
        values.append(compensation(slope[subject] * grid, expected))
    return remove_outliers(np.array(values), 2.5)


def welch(udl, random):
    """Two-sided Welch t-test with its Satterthwaite df and 95% CI on the difference."""
    variance = np.var(udl, ddof=1) / len(udl) + np.var(random, ddof=1) / len(random)
    df = variance ** 2 / ((np.var(udl, ddof=1) / len(udl)) ** 2 / (len(udl) - 1)
                          + (np.var(random, ddof=1) / len(random)) ** 2 / (len(random) - 1))
    test = ttest_ind(udl, random, equal_var=False)
    half_width = t.ppf(.975, df) * np.sqrt(variance)
    return test.statistic, df, test.pvalue, np.mean(udl) - np.mean(random), half_width


# ----------------------------------------------------------------- figure
def plot(results, random, udl):
    style()
    fig = plt.figure(figsize=(11.2, 7.2))
    grid = fig.add_gridspec(2, 3, hspace=.55, wspace=.55, left=.08, right=.985, bottom=.095, top=.95)
    bars = dict(color="k", lw=1, capsize=0, ls="none")

    ax = fig.add_subplot(grid[0, 0]); label(ax, "c")
    means, errors = [np.mean(random), np.mean(udl)], [sem(random), sem(udl)]
    ax.bar([0, 1], means, .55, color=["#9a9d9a", "#e6e6e6"], edgecolor="k", lw=1)
    ax.errorbar([0, 1], means, errors, **bars)
    rng = np.random.default_rng(3)
    for position, values in [(0, random), (1, udl)]:
        ax.scatter(rng.normal(position, .07, len(values)), values, s=12, color=".35", alpha=.32, linewidths=0)
    ax.set(xticks=[0, 1], xticklabels=["Random", "UDL"], xlim=(-.55, 1.55), ylim=(0, 70),
           yticks=[0, 20, 40, 60], xlabel="Condition", ylabel="% of compensation")

    ax = fig.add_subplot(grid[0, 1]); label(ax, "d")
    # Ten lags are computed; the panel shows the first nine, as in the paper.
    shown_lags, width = 9, .25
    lags = np.arange(1, shown_lags + 1)
    for offset, condition in enumerate(ORDER):
        x = lags + (offset - 1) * width
        ax.bar(x, results[condition]["group_index"][:shown_lags], width,
               color=COLOUR[condition], edgecolor="none")
        ax.errorbar(x, results[condition]["group_index"][:shown_lags],
                    results[condition]["group_index_sem"][:shown_lags], **bars)
    ax.axhline(0, color=".3", lw=.6)
    ax.set(xlim=(.4, 9.6), ylim=(-1, 4.5), xticks=[1, 4, 7], yticks=[0, 2, 4],
           xlabel="n-back", ylabel="SE Index (°)")

    for position, letter, key, y_limits, y_ticks in [
        (grid[0, 2], "e", "retention", (0, .9), [.2, .4, .6, .8]),
        (grid[1, 2], "h", "index_2to5", (0, 1.5), [.2, .6, 1., 1.4]),
    ]:
        ax = fig.add_subplot(position); label(ax, letter)
        for index, condition in enumerate(ORDER):
            ax.bar(index, results[condition][key], .45, color=COLOUR[condition], edgecolor="none")
            ax.errorbar(index, results[condition][key], results[condition][f"{key}_sem"], **bars)
        ax.set(xticks=range(3), xticklabels=[PSWITCH[c] for c in ORDER], xlim=(-.55, 2.55),
               ylim=y_limits, yticks=y_ticks, xlabel="P(switch)",
               ylabel="Retention" if letter == "e" else "SE Index (°)")

    for column, letter, key, title, y_limits, y_ticks in [
        (0, "f", "function_1back", "1-back", (-2.35, 2.35), [-2, -1, 0, 1, 2]),
        (1, "g", "function_2to5", "2~5-back", (-1.15, 1.15), [-1, 0, 1]),
    ]:
        ax = fig.add_subplot(grid[1, column]); label(ax, letter)
        for condition in ORDER:
            y, y_sem = results[condition][key], results[condition][f"{key}_sem"]
            shown = np.abs(BINS) <= 150
            ax.fill_between(BINS[shown], (y - y_sem)[shown], (y + y_sem)[shown],
                            color=COLOUR[condition], alpha=.18, lw=0)
            ax.plot(BINS[shown], y[shown], color=COLOUR[condition], lw=1.7)
        ax.axhline(0, color=".25", lw=.6, ls=(0, (9, 6)))
        ax.axvline(0, color=".25", lw=.6, ls=(0, (9, 6)))
        ax.set(xlim=(-165, 165), ylim=y_limits, xticks=[-90, 0, 90], yticks=y_ticks,
               xlabel="ΔTarget (°)", ylabel="Deviation (°)", title=title)
    return fig


def main():
    data = load("exp3")
    results = {condition: analyse_condition(data, condition) for condition in ORDER}
    for condition in ORDER:
        result = results[condition]
        print(f"Exp 3 {condition:6s}: N={result['n']:2d}, "
              f"1-back={result['group_index'][0]:.2f}±{result['group_index_sem'][0]:.2f} "
              f"(curve N={result['n_curve']}), "
              f"retention={result['retention']:.3f}±{result['retention_sem']:.3f}, "
              f"2-5-back={result['index_2to5']:.2f}±{result['index_2to5_sem']:.2f}")

    random, udl = compensation_random(), compensation_udl()
    random = random[np.isfinite(random)]
    for name, values in [("Random", random), ("UDL", udl)]:
        print(f"Figure 6c {name:6s}: N={len(values):2d}, "
              f"compensation={np.mean(values):.1f}±{sem(values):.1f}%")
    statistic, df, p, difference, half_width = welch(udl, random)
    print(f"Figure 6c Welch t({df:.1f}) = {statistic:.2f}, p = {p:.4g}, "
          f"UDL - Random = {difference:+.1f}% [95% CI {difference - half_width:+.1f}, "
          f"{difference + half_width:+.1f}]")

    show(plot(results, random, udl), "Exp3_fig6")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom")
    main()
