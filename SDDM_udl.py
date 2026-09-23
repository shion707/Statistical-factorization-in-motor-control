"""Figure 4a-b: bimodal reaching error under use-dependent learning, data beside model.

Data: ``data/udl_reaching.csv`` (Tsay et al. 2022, Experiments 1 and 2), already aligned to
the trained direction.  Model: the population drift-diffusion model of Figure 3.

ONE PARAMETER SET.  Panels a and b are the same simulation.  Nothing is re-tuned between
them; the only things that change are experimental:

    free response (Exp 1)    -> the decision boundary collapses
    delayed response (Exp 2) -> it does not
    probe distance           -> where the evidence bump is put

Panel a is simply the 90 degree probe of the free condition, so it is one slice of panel b.

"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import common

# --------------------------------------------------------------- parameters
# Every value below is imported from the Experiment 1-2 model.
from SDDM import (LR, SD2, SD, VAL_CAP, FORGET, BOUND, RATE, NOISE,      # noqa: F401
                  UNITS, DIRECTION, DEGREES, CEILING, value_field, wrap)

# The one difference from Experiment 1-2.  There, a free response has a boundary that
# collapses at a single fixed rate.  Here the rate varies from trial to trial: a Beta
# distribution rescaled to the mean below.   A delayed response does not collapse at all.
COLLAPSE = 3 * RATE              # mean collapse rate per step
BETA = (10, 50)                  # Beta shape of the trial-to-trial spread (CV = 0.29)
PRIOR = 1                        # the prior is a single repeated direction, one degree wide
MAX_STEPS = 4000                 # safety bound on the integration, not a model parameter;
                                 # every simulated trial crosses the boundary long before it

PROBE = -90                      # probe target for panel a, degrees from trained
SPLIT = -45                      # error below this counts as a habitual reach
TARGETS = [0, 30, 60, 90]        # probe distances in panel b
LEVELS = TARGETS                 # the same distances, for the data side
TEST_BLOCK = 3                   # blocks 1-2 are baseline; the habit starts at 3
SEED = 42
N_PANEL_A = 500
N_PANEL_B = 1500

ACCURATE, HABITUAL = "#5a5a5a", "#d95319"
FREE, DELAY = common.COLOUR[180], common.PALE[180]       # strong = free, pale = delayed


# ----------------------------------------------------------------- model
def collapse_rates(response, n, rng):
    """One boundary-decay rate per trial: a scaled Beta if free, zero if delayed."""
    if response == "Delay":
        return np.zeros(n)
    a, b = BETA
    return rng.beta(a, b, n) * (COLLAPSE * (a + b) / a)


def race(target, field, collapse, rng):
    """Accumulate until the population peak reaches the collapsing boundary.

    Returns the crossing time in steps -- interpolated between steps, so the reaction time
    is continuous -- and the signed error of the winning unit, in degrees.

    """
    n = collapse.size
    evidence = (np.exp(SD * (np.cos(DIRECTION - np.radians(target)) - 1))
                * RATE).astype(np.float32)
    base = field.astype(np.float32)

    step = np.arange(1.0, MAX_STEPS + 1)
    reach = base.max() + step * evidence.max() + 5 * np.sqrt(step) * CEILING
    early = np.flatnonzero(reach >= BOUND - collapse.max() * step)
    start = max(1.0, step[early[0]] - 1.0) if early.size else 1.0

    live = np.arange(n)
    decay = collapse.astype(np.float32)
    activity = (base[None, :] + np.float32(start) * evidence[None, :]
                + rng.standard_normal((n, UNITS), dtype=np.float32)
                * np.float32(NOISE * np.sqrt(start)))
    gap = activity.max(axis=1) - (BOUND - decay * np.float32(start))

    crossing = np.full(n, float(MAX_STEPS))
    error = np.zeros(n)
    t = start
    while live.size and t < MAX_STEPS:
        t += 1.0
        activity += (evidence[None, :] + rng.standard_normal(activity.shape, dtype=np.float32)
                     * np.float32(NOISE))
        now = activity.max(axis=1) - (BOUND - decay * np.float32(t))
        hit = now >= 0
        if hit.any():
            crossing[live[hit]] = t - now[hit] / np.maximum(now[hit] - gap[hit], 1e-30)
            error[live[hit]] = DEGREES[activity[hit].argmax(axis=1)] - target
            keep = ~hit
            live, activity, decay, gap = live[keep], activity[keep], decay[keep], now[keep]
        else:
            gap = now
    if live.size:
        error[live] = DEGREES[activity.argmax(axis=1)] - target
    return crossing, wrap(error)


def simulate(rng):
    """Every condition of both panels, from the one parameter set."""
    field = value_field(PRIOR)
    out = {}
    for response in ("Free", "Delay"):
        means, sems, trials = [], [], {}
        for target in TARGETS:
            n = N_PANEL_A if (response == "Free" and target == abs(PROBE)) else N_PANEL_B
            crossing, error = race(target, field, collapse_rates(response, n, rng), rng)
            trials[target] = (crossing, error)
            means.append(-error.mean())
            sems.append(error.std(ddof=1) / np.sqrt(n))
        out[response] = (np.array(means), np.array(sems))
        out[response + "_trials"] = trials
    # Panel a is the 90 degree probe of the free condition.  The probe sits at a negative
    # angle in the figure's convention, and the simulation puts it at a positive one.
    crossing, error = out["Free_trials"][abs(PROBE)]
    return out, error, crossing


# ----------------------------------------------------------------- data
def observed():
    """Panel a's probe trials and panel b's bias curves, from Tsay et al. 2022.
    """
    table = common.load("udl_reaching")
    table = table.assign(distance=table.relative_target_deg.abs())
    test = table[table.block >= TEST_BLOCK]

    # Panel a: probe trials from the test phase of the free-response experiment.
    free = test[(test.experiment == "E1") & (test.rt_s <= 0.7)]
    probe = free[free.distance == abs(PROBE)]

    def curve(experiment):
        """Test-phase bias minus the same participant's baseline, per probe distance."""
        rows = table[table.experiment == experiment]
        people = np.sort(rows.participant.unique())
        shift = np.full((len(people), len(LEVELS)), np.nan)
        for i, person in enumerate(people):
            one = rows[rows.participant == person]
            for j, level in enumerate(LEVELS):
                at = one[one.distance == level]
                shift[i, j] = (at[at.block >= TEST_BLOCK].aligned_error_deg.mean()
                               - at[at.block < TEST_BLOCK].aligned_error_deg.mean())
        n = np.isfinite(shift).sum(axis=0)
        return (-np.nanmean(shift, axis=0),
                np.nanstd(shift, axis=0, ddof=1) / np.sqrt(n))

    everything = test[(test.experiment == "E1") & (test.distance == abs(PROBE))]
    return dict(rt=probe.rt_s.to_numpy(), error=probe.aligned_error_deg.to_numpy(),
                rt_all=everything.rt_s.to_numpy(),
                error_all=everything.aligned_error_deg.to_numpy(),
                Free=curve("E1"), Delay=curve("E2"))


# ----------------------------------------------------------------- figure
def to_seconds(steps, reference):
    """Model time is in accumulation steps; stretch it onto the measured RT axis."""
    low, high = np.percentile(steps, [10, 90])
    lo, hi = np.percentile(reference, [10, 90])
    return (steps - low) / (high - low) * (hi - lo) + lo


def joint_panel(fig, cell, rt, error, title):
    """Scatter of error against RT with a histogram on each margin, split at SPLIT."""
    grid = cell.subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                            hspace=.06, wspace=.06)
    main = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=main)
    side = fig.add_subplot(grid[1, 1], sharey=main)

    split = [(error >= SPLIT, ACCURATE), (error < SPLIT, HABITUAL)]
    for mask, colour in split:
        main.scatter(rt[mask], error[mask], s=7, color=colour, alpha=.45, lw=0)
        top.hist(rt[mask], bins=np.arange(.05, .72, .012), color=colour, alpha=.85, lw=0)
        side.hist(error[mask], bins=np.arange(-100, 51, 2), orientation="horizontal",
                  color=colour, alpha=.85, lw=0)

    # Least-squares line of RT on error, drawn with the axes swapped.
    slope, intercept = np.polyfit(error, rt, 1)
    span = np.array([error.min(), error.max()])
    main.plot(slope * span + intercept, span, "--", color=".35", lw=1.6)
    for level in (0, PROBE):
        main.axhline(level, color="k", lw=.8, ls="--")
    main.set(xlim=(.05, .72), ylim=(-100, 50), xlabel="RT (s)", ylabel="Error (°)",
             xticks=[.2, .4, .6], yticks=[-100, -50, 0, 50])
    for margin in (top, side):
        margin.axis("off")
    top.set_title(title)
    return main


def make_figure(data, model_error, model_rt, model_bias):
    common.style()
    plt.rcParams["axes.titleweight"] = "bold"        # the Data / Model panel headings
    fig = plt.figure(figsize=(10, 9.4))
    outer = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=.34, wspace=.30,
                             left=.09, right=.97, top=.915, bottom=.135)

    left = joint_panel(fig, outer[0, 0], data["rt"], data["error"], "Data")
    joint_panel(fig, outer[0, 1], to_seconds(model_rt, data["rt"]), model_error, "Model")
    common.label(left, "a", x=-.30, y=1.34)
    left.text(.055, 3, "Probe\nTarget", fontsize=7.5, va="bottom")
    left.text(.42, PROBE + 3, "Repeat Target", fontsize=7.5, va="bottom")

    for column, (source, title) in enumerate([(data, "Data"), (model_bias, "Model")]):
        ax = fig.add_subplot(outer[1, column])
        for response, colour in (("Free", FREE), ("Delay", DELAY)):
            mean, sem = source[response]
            ax.errorbar(TARGETS, mean - mean[0], sem, color=colour, lw=2, capsize=0,
                        elinewidth=1.4)
        ax.axhline(0, color="k", lw=.8, ls="--")
        ax.set(xlim=(-8, 100), ylim=(-5, 36), xticks=[0, 40, 80], yticks=[0, 10, 20, 30],
               xlabel="Target (°)", ylabel="Bias (°)", title=title)
        if column == 0:
            common.label(ax, "b", x=-.24, y=1.10)
            ax.legend(handles=[Line2D([], [], color=FREE, lw=2, label="Free"),
                               Line2D([], [], color=DELAY, lw=2, label="Delay")],
                      frameon=False, fontsize=8.5, loc="upper left")

    fig.suptitle("Figure 4a–b — bimodal error under use-dependent learning "
                 "(Tsay et al. 2022 data, population drift-diffusion model)",
                 fontsize=11.5, y=.972)
    fig.text(.5, .055, f"a: probe {abs(PROBE)}° from the trained direction; grey = accurate "
             f"reaches, orange = habitual reaches (error < {SPLIT}°)",
             ha="center", fontsize=7.5, color=".4")
    fig.text(.5, .028, "one parameter set throughout; model RT is in accumulation steps, "
             "stretched onto the measured RT axis for display only",
             ha="center", fontsize=7.5, color=".45")
    return fig


# ----------------------------------------------------------------- report
STATISTICS = [("habitual reaches (%)", "8.0f", lambda rt, e: (e < SPLIT).mean() * 100),
              ("error-RT correlation", "8.2f", lambda rt, e: np.corrcoef(rt, e)[0, 1]),
              ("accurate mode (°)", "8.1f", lambda rt, e: np.median(e[e >= SPLIT])),
              ("habitual mode (°)", "8.1f", lambda rt, e: np.median(e[e < SPLIT])),
              ("habitual RT / accurate RT", "8.2f",
               lambda rt, e: rt[e < SPLIT].mean() / rt[e >= SPLIT].mean()),
              ("between the modes (%)", "8.0f",
               lambda rt, e: ((e > -70) & (e < -20)).mean() * 100)]


def report(data, model_error, model_rt, model_bias):
    """Print summary statistics of panel a, data beside model, and the panel b curves.

    The data appear twice: the free-response probe trials with RT <= 0.7 s, which the
    figure plots, and all of them.
    """
    print(f"\n{'':>26s} {'data<=.7s':>10s} {'data all':>9s} {'model':>8s}")
    for name, fmt, statistic in STATISTICS:
        print(f"{name:>26s} {statistic(data['rt'], data['error']):{fmt}}  "
              f"{statistic(data['rt_all'], data['error_all']):{fmt}} "
              f"{statistic(model_rt, model_error):{fmt}}")

    print(f"\n{'bias (°)':>10s} " + " ".join(f"{t:>7d}°" for t in TARGETS)
          + "     (referenced to the 0° probe, as plotted)")
    for response in ("Free", "Delay"):
        for name, values in ((f"{response} data", data[response][0]),
                             (f"{response} model", model_bias[response][0])):
            print(f"{name:>10s} " + " ".join(f"{v - values[0]:8.1f}" for v in values))


def main():
    rng = np.random.default_rng(SEED)
    data = observed()
    model_bias, model_error, model_rt = simulate(rng)

    report(data, model_error, model_rt, model_bias)
    common.show(make_figure(data, model_error, model_rt, model_bias), "SDDM_udl")


if __name__ == "__main__":
    main()
