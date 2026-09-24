"""Figure 5a-f: what a repulsive sequential effect does to the central tendency.

Panels a-c take three fixed strengths -- none, the strength observed in Exp 1, and twice
that -- and show the two ingredients, their sum, and the error that results.  Panels d-f
sweep the strength continuously: how it trades bias against variance, where the error is
minimised under each prior, and how one fixed effect is distorted when it is measured
under a narrower prior.

"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

import common

BIN = 20
STRENGTH_GRID = np.arange(0, 24.02, .05)
STRENGTH_SWEEP = np.arange(0, 15.02, .25)
CENTRES = np.arange(-9, 10) * BIN

SIGMA = 36.0            # width of the derivative-of-Gaussian kernel, degrees

# prior -> (data condition, bias slope measured by Exp1_2.py, polynomial order used when the
# simulated responses are analysed like the data in panel f)
CONDITIONS = {"Wide": ("d180", 0.040, 20), "Medium": ("d120", 0.059, 15),
              "Narrow": ("d60", 0.113, 12)}
COLOUR = {"Wide": "#2e9e8f", "Medium": "#e0821e", "Narrow": "#3b5a9a"}

# Panels a-c compare three fixed strengths of the sequential effect; d-f sweep it.
OBSERVED = 7.9                                   # the strength observed in Exp 1 (peak 1.47°)
STRENGTHS = (0.0, OBSERVED, 2 * OBSERVED)
STRENGTH_LABEL = ("no SE", "Observed SE", "2 × Observed SE")
STRENGTH_COLOUR = ("#f4b8c0", "#c2559b", "#5c3b9c")


def kernel(delta, sigma):
    """The sequential-effect kernel at strength 1: 1000 times the derivative of a Gaussian
    density with SD `sigma`, signed so that a positive strength is repulsive.  `delta` is the
    current target minus the previous one."""
    return 1000 * delta / (np.sqrt(2 * np.pi) * sigma ** 3) * np.exp(-(delta ** 2) / (2 * sigma ** 2))


def unit_peak(sigma):
    """Peak of the kernel at strength 1, in degrees, reached at |delta| = sigma."""
    return 1000 * np.exp(-0.5) / (np.sqrt(2 * np.pi) * sigma ** 2)


def se_function(target_index, error, order):
    """The sequential-effect function, measured as in the data"""
    dev = common.remove_outliers_2d(common.detrend(target_index, error, order), 3)
    change = np.full_like(target_index, np.nan, dtype=float)
    change[1:] = target_index[:-1] - target_index[1:]
    binned = common.round_half_away(change / BIN)
    out = np.full((target_index.shape[1], len(CENTRES)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for column in range(target_index.shape[1]):
            for j, edge in enumerate(CENTRES / BIN):
                out[column, j] = np.nanmean(
                    common.remove_outliers(dev[:, column][binned[:, column] == edge], 2.5))
            out[column] -= np.nanmean(out[column])
    return out


def load(condition):
    rows = common.load("exp1_2")
    rows = rows[rows["condition"] == condition]
    target, _ = common.matrix(rows, "target_deg")
    index, _ = common.matrix(rows, "target_index")
    error, _ = common.matrix(rows, "error_deg")
    change = np.full_like(target, np.nan)
    change[1:] = target[1:] - target[:-1]
    return target, index, error, change


def simulate(target, change, slope, strength, sigma):
    """Responses under this central tendency and this much sequential effect."""
    return target * (1 - slope) + strength * kernel(change, sigma)


def analyse(sigma):
    grid = STRENGTH_GRID
    out = {}
    for name, (condition, slope, order) in CONDITIONS.items():
        target, index, error, change = load(condition)
        ok = np.isfinite(target) & np.isfinite(change)
        bias_error, effect = slope * target[ok], kernel(change[ok], sigma)
        curve = np.sqrt(np.mean((bias_error[None, :] - grid[:, None] * effect[None, :]) ** 2, 1))
        out[name] = dict(strength=STRENGTH_GRID, curve=curve, target=target, index=index,
                         change=change, slope=slope, order=order)
    return out


def bias_and_variance(entry, strength, sigma):
    response = simulate(entry["target"], entry["change"], entry["slope"], strength, sigma)
    targets = np.unique(entry["target"][np.isfinite(entry["target"])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.array([np.nanmean(response[entry["target"] == t]) for t in targets])
        var = np.array([np.nanvar(response[entry["target"] == t], ddof=1) for t in targets])
    return targets, targets - mean, np.nanmean(np.abs(targets - mean)), np.nanmean(var)


def recovered(sigma, results):
    """Panel f: the same constant kernel, measured in each prior condition.

    The kernel is drawn at the observed strength, the same one panels a-e call "Observed
    SE", so panel f recovers the effect that panel e marks as the operating point.
    """
    out = {}
    for name, (condition, slope, order) in CONDITIONS.items():
        entry = results[name]
        response = simulate(entry["target"], entry["change"], slope, OBSERVED, sigma)
        measured = se_function(entry["index"], response - entry["target"], order)
        with warnings.catch_warnings():          # bins wider than the range are empty
            warnings.simplefilter("ignore", RuntimeWarning)
            out[name] = np.nanmean(measured, 0)
    return out


def make_figure(sigma, results, recovered_functions):
    common.style()
    fig = plt.figure(figsize=(11.4, 6.8))
    # Two rows of unequal panel counts, so each gets its own subfigure and its own spacing.
    top, bottom = fig.subfigures(2, 1, hspace=.02)
    upper = top.subplots(1, 4)
    lower = bottom.subplots(1, 3)
    top.subplots_adjust(wspace=.62, left=.065, right=.985, top=.86, bottom=.20)
    bottom.subplots_adjust(wspace=.52, left=.065, right=.985, top=.90, bottom=.22)
    wide = results["Wide"]
    slope = CONDITIONS["Wide"][1]

    # a. the two ingredients, drawn on their own: a linear attractive bias toward the
    # centre of the prior, and the repulsive kernel at each of the three strengths.
    ax = upper[0]
    common.label(ax, "a", x=-.20)
    span = np.linspace(-90, 90, 200)
    ax.plot(span, slope * span, color=STRENGTH_COLOUR[2], lw=2)
    ax.set(xlim=(-95, 95), ylim=(-4.6, 4.6), xticks=[], yticks=[],
           xlabel="Target", ylabel="Bias", title="Attractive bias")

    ax = upper[1]
    delta = np.linspace(-180, 180, 400)
    for strength, colour, name in zip(STRENGTHS[::-1], STRENGTH_COLOUR[::-1],
                                      STRENGTH_LABEL[::-1]):
        ax.plot(delta, strength * kernel(delta, sigma),
                color=colour, lw=2, label=name)
    ax.set(xlim=(-180, 180), ylim=(-4.3, 4.3), xticks=[], yticks=[],
           xlabel="ΔTarget", ylabel="Deviation", title="Repulsive effect")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.1)

    # b. the two summed: a stronger kernel flattens the central tendency
    ax = upper[2]
    common.label(ax, "b", x=-.28)
    for strength, colour in zip(STRENGTHS, STRENGTH_COLOUR):
        targets, bias, _, _ = bias_and_variance(wide, strength, sigma)
        ax.plot(targets, bias, color=colour, lw=2)
    ax.axhline(0, color="k", lw=.6, ls="--")
    ax.axvline(0, color="k", lw=.6, ls="--")
    ax.set(xlim=(-95, 95), ylim=(-4.6, 4.6), xticks=[-80, -40, 0, 40, 80],
           yticks=[-4, -2, 0, 2, 4], xlabel="Target (°)", ylabel="Bias (°)")

    # c. error is lowest at the observed strength, not at zero and not at twice it.  The
    # three bars differ by about a fifth of a degree, so the axis is cropped to that range.
    ax = upper[3]
    common.label(ax, "c", x=-.28)
    rmse = np.array([wide["curve"][np.argmin(np.abs(wide["strength"] - s))] for s in STRENGTHS])
    low, high = np.floor(rmse.min() * 10) / 10, np.ceil(rmse.max() * 10) / 10
    ax.bar(range(3), rmse, .6, color=STRENGTH_COLOUR, edgecolor="k", lw=.7)
    ax.set(xlim=(-.6, 2.6), ylim=(low, high), xticks=range(3),
           xticklabels=["0", "Obs.", "2 × Obs."], yticks=np.arange(low, high + .01, .2),
           xlabel="SE strength", ylabel="RMSE (°)")

    # d. bias falls and variance grows with the strength of the sequential effect
    ax = lower[0]
    common.label(ax, "d", x=-.17)
    measured = [bias_and_variance(results["Wide"], s, sigma) for s in STRENGTH_SWEEP]
    ax.plot(STRENGTH_SWEEP, [m[2] for m in measured], color="k", lw=2)
    twin = ax.twinx()
    twin.plot(STRENGTH_SWEEP, [m[3] for m in measured], color="#c0392b", lw=2)
    twin.spines["top"].set_visible(False)
    twin.tick_params(axis="y", colors="#c0392b")
    twin.set_ylabel("Variance (°²)", color="#c0392b")
    twin.set(ylim=(-.12, 2.08), yticks=[0, 1, 2])
    ax.set(xlim=(0, 15), ylim=(1.24, 1.84), xticks=[0, 5, 10, 15],
           yticks=[1.3, 1.5, 1.7], xlabel="SE strength", ylabel="Bias (°)")

    # e. the optimum moves outward as the prior narrows
    ax = lower[1]
    common.label(ax, "e", x=-.17)
    for name in CONDITIONS:
        entry = results[name]
        relative = ((entry["curve"] - entry["curve"].min())
                    / (entry["curve"][0] - entry["curve"].min()))
        ax.plot(entry["strength"], relative, color=COLOUR[name], lw=2)
    ax.set(xlim=(0, 24), ylim=(-.1, 1.45), xticks=[0, 5, 10, 15, 20], yticks=[0, .5, 1],
           xlabel="SE strength", ylabel="Relative RMSE")

    # f. one constant kernel, distorted differently in each condition
    ax = lower[2]
    common.label(ax, "f", x=-.17)
    keep = np.abs(CENTRES) <= 120
    for name in CONDITIONS:
        ax.plot(CENTRES[keep], recovered_functions[name][keep], color=COLOUR[name], lw=2)
    ax.set(xlim=(-125, 125), ylim=(-1.45, 1.45), xticks=[-90, 0, 90], yticks=[-1, 0, 1],
           xlabel="ΔTarget (°)", ylabel="Deviation (°)")

    return fig


def main():
    results = analyse(SIGMA)
    common.show(make_figure(SIGMA, results, recovered(SIGMA, results)),
                "SE_simulation")


if __name__ == "__main__":
    main()
