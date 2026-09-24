"""Figure 3b-g with the spatial drift-diffusion model drawn over the Experiment 2 data.

The data half of every panel is the same as in ``Exp1_2.py``.

THE MODEL.  A population of units tiles movement direction.  Two things drive it: a value
field left behind by experience with the target distribution, which pre-activates the
directions that occur often, and evidence about the current target, a von Mises bump added
at every time step.  The movement is committed once the most active unit reaches the
decision boundary, and is read out as that unit's preferred direction.  The value field is
still part of the total activity at that moment, so the read-out is pulled toward the
centre of the prior: the attractive bias.

Delayed and free responses differ only in the boundary.  A delayed response waits, so the
boundary is fixed and the evidence has time to dominate the field.  A free response may go
at once, modelled by a boundary that collapses over time, so commitment comes earlier while
the field still dominates, and the bias is larger.  Panels b, c and e show that difference.

Noise is handled analytically.  The boundary is lowered by ``CEILING * sqrt(t)``, the
expected largest excursion of accumulated noise across the population, so one noiseless
race lands at the mean crossing time.  Trial-to-trial spread comes from applying the
accumulated noise once at the moment of commitment and re-reading the winning unit.

"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from common import COLOUR, PALE, load, style, label, show
from Exp1_2 import CONDITIONS, SIMPLE_HALF, simple_rt, summarize


LR, SD2, SD = 9.7555556e-5, 1.1394, 3.3321  # learning rate, field width, evidence width
VAL_CAP, FORGET = 0.0727799792, 0.99       # ceiling of the value field, and the fraction of it
                                           # kept from one trial to the next.  
BOUND, RATE = 0.015, 0.00005               # decision boundary, evidence per step
NOISE = RATE * 1.4506                      # noise per step
COLLAPSE = 0.000089                        # boundary decay per step, free responses only

UNITS, MAX_STEPS = 2000, 1000
CURVE_POINTS, METRIC_STEP, SAMPLES, SEED = 150, 5, 500, 99
REFINE = 8                       # sub-steps per time step; the crossing time converges by 8
PRIORS = [180, 120, 60]                              # prior range width in degrees

RT_STEP_MS = 6.0954             # duration of one time step, fitted to the free-response mean RTs
NONDECISION_MS = 0.0            # no separate non-decision time is added

DIRECTION = np.linspace(-np.pi, np.pi, UNITS + 1)[:-1]
DEGREES = np.degrees(DIRECTION)
CEILING = NOISE * np.sqrt(2 * np.log(UNITS))


# ----------------------------------------------------------------- the model
def wrap(degrees):
    """Signed angular difference in degrees.  Local to the model: unlike the one-step
    ``common.wrap180`` of the data analyses, this is a full modular wrap."""
    return (degrees + 180) % 360 - 180


def value_field(prior):
    """What repeated exposure to a uniform prior of the given width leaves behind.

    ``prior`` is the width of the uniform target range in degrees, so a width of 1 is the
    limiting case of a single direction repeated over and over.
    """
    targets = np.radians(np.arange(prior) - prior / 2)
    drive = sum(np.exp(SD2 * np.cos(DIRECTION - t)) for t in targets)
    drive *= 1 / (prior * 2 * np.pi * np.i0(SD2))  # E(u): von Mises density averaged over the prior
    field = LR * drive / ((1 - FORGET) + LR * drive / VAL_CAP)
    return field + field.max() * 0.5           # units are never completely silent


def race(targets, prior, collapsing, rng=None):
    """Race the population to the boundary once per target.

    Returns bias toward the centre, crossing time in steps, and trial-to-trial spread.
    Spread is only computed when an ``rng`` is given.

    The race is integrated REFINE times per time step.  The dynamics are unchanged --
    evidence and boundary collapse are divided by REFINE and the crossing time is reported
    back in whole time steps.
    """
    field = value_field(prior)
    collapse = COLLAPSE if collapsing else 0.0
    bias, crossing, spread = (np.full(len(targets), np.nan) for _ in range(3))

    for i, target in enumerate(targets):
        evidence = np.exp(SD * (np.cos(DIRECTION - np.radians(target)) - 1)) * RATE / REFINE
        activity, sub = field, 0
        while sub < MAX_STEPS * REFINE:
            elapsed = sub / REFINE
            if activity.max() >= BOUND - collapse * elapsed - CEILING * np.sqrt(elapsed):
                break
            sub += 1
            activity = activity + evidence
        elapsed = sub / REFINE
        bias[i] = -wrap(DEGREES[activity.argmax()] - target)
        crossing[i] = elapsed
        if rng is not None:
            jitter = rng.standard_normal((SAMPLES, UNITS)) * NOISE * np.sqrt(max(elapsed, 1.0))
            spread[i] = wrap(DEGREES[(activity + jitter).argmax(axis=1)] - target).std()
    return bias, crossing, spread


def simulate():
    """Run all six conditions, in the order of ``CONDITIONS`` so the seed is reproducible."""
    rng = np.random.default_rng(SEED)
    model = {}
    for condition, (response, prior, half) in CONDITIONS.items():
        collapsing = response == "Free"
        curve = np.linspace(-half, half, CURVE_POINTS)
        bias, crossing, _ = race(curve, prior, collapsing)
        # Spread is expensive, so it is measured on a coarser target grid.
        _, _, spread = race(np.arange(-half, half + 1, METRIC_STEP), prior, collapsing, rng)
        model[condition] = dict(target=curve, bias=bias, rt_steps=crossing,
                                slope=np.polyfit(curve, bias, 1)[0],
                                rt_mean=crossing.mean(), sd=spread.mean())
    return model


def apply_calibration(model):
    """Put the model's reaction time on a millisecond axis.
    """
    for entry in model.values():
        entry["rt_ms"] = entry["rt_steps"] * RT_STEP_MS + NONDECISION_MS
        entry["rt_mean_ms"] = entry["rt_mean"] * RT_STEP_MS + NONDECISION_MS


# ----------------------------------------------------------------- figure
def smooth(x, y, order):
    """Polynomial smoothing of a model curve."""
    grid = np.linspace(x.min(), x.max(), 200)
    return grid, np.polyval(np.polyfit(x, y, order), grid)


def band(ax, x, mean, error, colour, alpha):
    """One data curve with its standard-error band."""
    ax.fill_between(x, mean - error, mean + error, color=colour, alpha=alpha, lw=0)
    ax.plot(x, mean, color=colour, lw=2)


def bar(ax, x, width, value, error, colour, prediction):
    """One data bar with its error bar, and the model's value as a white diamond."""
    ax.bar(x, value, width, color=colour, edgecolor="k", lw=.7)
    ax.errorbar(x, value, error, color="k", lw=1.1, capsize=0)
    ax.scatter(x, prediction, 70, marker="D", color="w", edgecolor="k", lw=1.1, zorder=5)


def figure(data, model, simple):
    fig = plt.figure(figsize=(11, 10.6))
    grid = fig.add_gridspec(3, 3, hspace=.46, wspace=.36, left=.07, right=.985,
                            top=.93, bottom=.10, height_ratios=[1, 1, 1])

    # Rows 0 and 1 hold panels b, c and d: the model on top, the data underneath, as in
    # the paper.
    for column, (letter, prefix, title) in enumerate([
            ("b", "d", "Delay Response"), ("c", "f", "Free Response")]):
        top = fig.add_subplot(grid[0, column])
        label(top, letter)
        for prior in PRIORS:
            key, colour = f"{prefix}{prior}", COLOUR[prior]
            top.plot(*smooth(model[key]["target"], model[key]["bias"], 8), color=colour, lw=2)
        top.set_title(title, fontsize=10, fontweight="bold")

        bottom = fig.add_subplot(grid[1, column])
        for prior in PRIORS:
            key, colour = f"{prefix}{prior}", COLOUR[prior]
            band(bottom, data[key]["target"], data[key]["bias"], data[key]["bias_se"], colour, .22)

        for ax in (top, bottom):
            ax.axhline(0, color="k", lw=.6, ls="--")
            ax.axvline(0, color="k", lw=.6, ls="--")
            ax.set(xlim=(-95, 95), ylim=(-7, 7), xticks=[-80, -40, 0, 40, 80],
                   yticks=[-6, -3, 0, 3, 6], xlabel="Target (°)", ylabel="Bias (°)")

    # Panel d: free-response reaction time, model above and data below.
    top = fig.add_subplot(grid[0, 2])
    label(top, "d")
    for prior in PRIORS:
        top.plot(*smooth(model[f"f{prior}"]["target"], model[f"f{prior}"]["rt_ms"], 6),
                 color=COLOUR[prior], lw=2)
    top.set_title("Free Response", fontsize=10, fontweight="bold")

    bottom = fig.add_subplot(grid[1, 2])
    for prior in PRIORS:
        key, colour = f"f{prior}", COLOUR[prior]
        band(bottom, data[key]["target"], data[key]["rt_mean"], data[key]["rt_se"], colour, .20)
    for key, half in (("s120", 60), ("s60", 30)):
        target, mean, error = simple[key]
        colour = COLOUR[half * 2]
        # Fill only: band() would also draw a solid line underneath the dashed baseline.
        bottom.fill_between(target, mean - error, mean + error, color=colour, alpha=.14, lw=0)
        bottom.plot(target, mean, color=colour, lw=1.6, ls=(0, (3, 2)))
    bottom.text(0, 308, "Simple RT", ha="center", fontsize=8, color=".35")

    for ax in (top, bottom):
        ax.set(xlim=(-95, 95), ylim=(300, 460), xticks=[-80, -40, 0, 40, 80],
               yticks=[300, 350, 400, 450], xlabel="Target (°)", ylabel="RT (ms)")

    # Panels e, f and g: one bar per condition, grouped by prior range, with the model's
    # prediction as a diamond.  Each series is (offset, condition prefix, palette, width).
    delayed, free = (-.18, "d", PALE, .36), (.18, "f", COLOUR, .36)
    solo = (0, "f", COLOUR, .6)                # panel g shows free responses only
    for letter, position, value, prediction, series, axis_label, limits, ticks in [
            ("e", (2, 0), "slope", "slope", (delayed, free), "Slope", (0, .32), [0, .1, .2, .3]),
            ("f", (2, 1), "sd", "sd", (delayed, free), "Standard Deviation (°)",
             (0, 8.5), [0, 2, 4, 6, 8]),
            ("g", (2, 2), "rt_avg", "rt_mean_ms", (solo,), "RT (ms)",
             (300, 460), [300, 350, 400, 450])]:
        ax = fig.add_subplot(grid[position])
        label(ax, letter)
        for i, prior in enumerate(PRIORS):
            for offset, prefix, palette, width in series:
                key = f"{prefix}{prior}"
                bar(ax, i + offset, width, data[key][value], data[key][value + "_se"],
                    palette[prior], model[key][prediction])
        ax.set(xticks=[0, 1, 2], xticklabels=["180", "120", "60"], xlabel="Prior Range (°)",
               ylabel=axis_label, ylim=limits, yticks=ticks)
        if letter == "e":
            ax.legend(handles=[Patch(facecolor="0.75", edgecolor="k", label="Delayed"),
                               Patch(facecolor="0.35", edgecolor="k", label="Free"),
                               Line2D([], [], ls="", marker="D", mfc="w", mec="k", label="Model")],
                      frameon=False, fontsize=8, loc="upper left", handlelength=1.2)

    fig.suptitle("Figure 3 — spatial drift-diffusion model over the data", fontsize=12, y=.975)
    fig.text(.5, .028, "b–d: model above, data below (mean ± s.e.m.)   ·   "
             "e–g: bars are data, ♦ is the model   |   "
             "colours: Wide=teal · Medium=orange · Narrow=navy",
             ha="center", fontsize=7.5, color=".4")
    fig.text(.5, .004, "calibrated, not predicted:  d and g map model steps to ms via the "
             "three free-response means", ha="center", fontsize=7.5, color=".45")
    return fig


# ----------------------------------------------------------------- entry point
def main():
    style()
    reaching = load("exp1_2")
    data = {key: summarize(reaching, key) for key in CONDITIONS}
    simple_table = load("Simple_rt")
    simple = {key: simple_rt(simple_table, key) for key in SIMPLE_HALF}
    model = simulate()
    apply_calibration(model)

    print(f"{'cond':>6s} {'slope':>16s} {'SD (deg)':>16s} {'RT (ms)':>17s}")
    print(f"{'':>6s} {'data':>7s} {'model':>8s} {'data':>7s} {'model':>8s} {'data':>8s} {'model':>8s}")
    for key, (response, _, _) in CONDITIONS.items():
        seen, fit = data[key], model[key]
        rt = (f"{seen['rt_avg']:8.0f} {fit['rt_mean_ms']:8.0f}" if response == "Free"
              else f"{'-':>8s} {'-':>8s}")     # delayed responses carry no reaction time
        print(f"{key:<6s} {seen['slope']:7.3f} {fit['slope']:8.3f} "
              f"{seen['sd']:7.2f} {fit['sd']:8.2f} {rt}")

    show(figure(data, model, simple), "SDDM")


if __name__ == "__main__":
    main()
