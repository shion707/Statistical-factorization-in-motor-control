"""Helpers shared by the analysis and simulation scripts.
"""
from pathlib import Path
import atexit
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Prior range in degrees -> colour.  Delayed responses use the pale shade, free the strong one.
COLOUR = {180: "#2e9e8f", 120: "#e0821e", 60: "#3b5a9a"}
PALE = {180: "#a9d8d0", 120: "#f3cd9a", 60: "#aab6d6"}


def load(name):
    """Read one table from data/.
    """
    return pd.read_csv(DATA / f"{name}.csv", na_values=["NAN"])


# ----------------------------------------------------------------- small statistics
def sem(x, axis=None):
    return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.sum(np.isfinite(x), axis=axis))


def nan_mean_sem(values):
    """Mean and SEM over the finite entries, quiet when a bin is empty."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=0), sem(values, axis=0)


def wrap180(v):
    """One-step wrap onto (-180, 180]"""
    v = np.asarray(v, dtype=float).copy()
    v[v > 180] -= 360
    v[v < -180] += 360
    return v


def round_half_away(x):
    """Half away from zero, unlike numpy's half to even."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.floor(x + .5), np.ceil(x - .5))


def remove_outliers(x, n_sd=2.5):
    """Blank anything beyond n_sd of the mean."""
    x = np.asarray(x, dtype=float).copy()
    if np.isfinite(x).sum() < 2:
        return x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, sd = np.nanmean(x), np.nanstd(x, ddof=1)
    if np.isfinite(sd) and sd > 0:
        x[np.abs(x - mean) > n_sd * sd] = np.nan
    return x


def remove_outliers_2d(x, n_sd=2.5):
    """Blank entries more than n_sd SD from their column mean or from their row mean."""
    x = np.asarray(x, dtype=float)
    clean = x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for axis in (0, 1):
            mean = np.nanmean(x, axis=axis, keepdims=True)
            sd = np.nanstd(x, axis=axis, ddof=1, keepdims=True)
            clean[np.abs(x - mean) > n_sd * sd] = np.nan
    return clean


def moving_mean(x, width=10):
    out = np.full_like(np.asarray(x, dtype=float), np.nan)
    before, after = width // 2, width // 2 - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for row in range(len(x)):
            out[row] = np.nanmean(x[max(0, row - before):min(len(x), row + after + 1)], axis=0)
    return out


def linear_slope(x, y):
    """Least-squares slope and its standard error"""
    ok = np.isfinite(y)
    design = np.column_stack((np.ones(ok.sum()), x[ok]))
    beta, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    residual = y[ok] - design @ beta
    covariance = (residual @ residual) / (ok.sum() - 2) * np.linalg.inv(design.T @ design)
    return beta[1], np.sqrt(covariance[1, 1])


def slope_per_column(x, y):
    """Least-squares slope of each column of ``y`` on ``x``, over the finite entries.

    Used to turn a target-by-participant matrix into one slope per participant, so a
    group slope can carry a between-subjects standard error like the other bar panels.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(y.shape[1], np.nan)
    for column in range(y.shape[1]):
        ok = np.isfinite(y[:, column])
        if ok.sum() >= 2:
            out[column] = np.polyfit(x[ok], y[ok, column], 1)[0]
    return out


# ----------------------------------------------------------------- reshaping
def matrix(data, value, row="trial"):
    participants = np.sort(data["participant"].unique())
    index = data[row].to_numpy(int)
    out = np.full((index.max(), len(participants)), np.nan)
    for column, participant in enumerate(participants):
        rows = data["participant"].to_numpy() == participant
        out[index[rows] - 1, column] = data[value].to_numpy(float)[rows]
    return out, participants


def target_means(data, value):
    """Target-by-participant matrix of per-target means, for the Experiment 1-2 tables."""
    participants = np.sort(data["participant"].unique())
    n = int(data["target_index"].max())
    out = np.full((n, len(participants)), np.nan)
    for column, participant in enumerate(participants):
        means = data.loc[data["participant"] == participant].groupby("target_index")[value].mean()
        out[means.index.to_numpy(int) - 1, column] = means.to_numpy(float)
    return np.arange(1, n + 1), participants, out


def detrend(target, error, order):
    """Residual after a per-participant polynomial fit of error on target.

    The fit is done in a standardised coordinate rather than in raw target units.  That is
    a change of basis only -- the fitted curve is the same polynomial -- but at these
    orders the Vandermonde matrix built on raw units is worse conditioned than float64 can
    represent, so the solver silently drops coefficients and the answer starts to depend on
    the LAPACK build instead of on the data.
    """
    residual = np.full_like(error, np.nan, dtype=float)
    for column in range(error.shape[1]):
        ok = np.isfinite(target[:, column]) & np.isfinite(error[:, column])
        if ok.sum() <= order:
            continue
        x = target[ok, column]
        standardised = (x - x.mean()) / x.std()
        fit = np.polyfit(standardised, error[ok, column], order)
        residual[ok, column] = error[ok, column] - np.polyval(fit, standardised)
    return residual


# ----------------------------------------------------------------- figures
def style():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
        "axes.titlesize": 10, "xtick.direction": "out", "ytick.direction": "out",
        "svg.fonttype": "none", "pdf.fonttype": 42})


def label(ax, letter, x=-.24, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontweight="bold", fontsize=13)


# Figures open in a window when the script finishes.  Set this to True to write PNG and
# PDF files as well, which is what you want on a machine with no display.
SAVE_FIGURES = False


def show(fig, name):
    """Name the figure's window, and write it to disk if SAVE_FIGURES is set.
    """
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title(name)
    if SAVE_FIGURES:
        for extension in ("png", "pdf"):
            fig.savefig(HERE / f"{name}.{extension}", dpi=150, bbox_inches="tight")
        print(f"Saved {name}.png and {name}.pdf")


@atexit.register
def _open_windows():
    if plt.get_fignums() and matplotlib.get_backend().lower() != "agg":
        plt.show()
