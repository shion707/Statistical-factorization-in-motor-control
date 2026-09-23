"""Statistical tests

Every per-participant measure is taken from the function the corresponding figure script
uses, so a number printed here and a bar in a figure always come from the same pipeline.

"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import common
import Exp1_2
import Exp3


ALPHA = 0.05
FREE = ["f180", "f120", "f60"]
DELAYED = ["d180", "d120", "d60"]
PRIORS = (180, 120, 60)


# ----------------------------------------------------------------- test machinery
def _design(levels_a, levels_b):
    """Dummy-coded design blocks for a two-factor between-subjects model."""
    a = pd.get_dummies(levels_a, drop_first=True).to_numpy(float)
    b = pd.get_dummies(levels_b, drop_first=True).to_numpy(float)
    interaction = np.column_stack([a[:, i] * b[:, j]
                                   for i in range(a.shape[1]) for j in range(b.shape[1])])
    return np.ones((len(a), 1)), a, b, interaction


def _rss(design, y):
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return residual @ residual


def factorial_anova(y, levels_a, levels_b):
    """Two-factor between-subjects ANOVA, Type II sums of squares.
    """
    y = np.asarray(y, dtype=float)
    one, a, b, ab = _design(levels_a, levels_b)
    full = np.hstack([one, a, b, ab])
    df_error = len(y) - full.shape[1]
    rss_full = _rss(full, y)
    mse = rss_full / df_error

    both_main = _rss(np.hstack([one, a, b]), y)
    extra = {"A": _rss(np.hstack([one, b]), y) - both_main,
             "B": _rss(np.hstack([one, a]), y) - both_main,
             "AB": both_main - rss_full}

    out = []
    for key, df_effect in (("A", a.shape[1]), ("B", b.shape[1]), ("AB", ab.shape[1])):
        ss = extra[key]
        f = (ss / df_effect) / mse
        out.append((key, f, df_effect, df_error, stats.f.sf(f, df_effect, df_error),
                    ss / (ss + rss_full)))
    return out


def one_way_anova(groups):
    """One-way ANOVA with partial eta squared."""
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    grand = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_between = len(groups) - 1
    df_within = sum(len(g) for g in groups) - len(groups)
    f = (ss_between / df_between) / (ss_within / df_within)
    return f, df_between, df_within, stats.f.sf(f, df_between, df_within), \
        ss_between / (ss_between + ss_within)


def welch_anova(groups):
    """Welch's ANOVA, for groups whose variances differ."""
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    mean = np.array([g.mean() for g in groups])
    weight = n / np.array([g.var(ddof=1) for g in groups])

    weighted_mean = (weight * mean).sum() / weight.sum()
    between = (weight * (mean - weighted_mean) ** 2).sum() / (k - 1)
    correction = ((1 - weight / weight.sum()) ** 2 / (n - 1)).sum()
    f = between / (1 + 2 * (k - 2) / (k ** 2 - 1) * correction)
    df_error = (k ** 2 - 1) / (3 * correction)
    return f, k - 1, df_error, stats.f.sf(f, k - 1, df_error)


def games_howell(groups, labels):
    """Games-Howell post-hoc comparisons, the companion to Welch's ANOVA."""
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    out = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            q = abs(a.mean() - b.mean()) / np.sqrt((va + vb) / 2)
            df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
            out.append((labels[i], labels[j], q, df,
                        stats.studentized_range.sf(q, len(groups), df)))
    return out


def levene(groups):
    """Levene's test on the group medians."""
    groups = [np.asarray(g, float) for g in groups]
    return stats.levene(*[g[np.isfinite(g)] for g in groups], center="median")


def permutation_retention(matrices, n_perm=4000, seed=0):
    """Do the three autocorrelation conditions differ in how fast the effect decays?
    Returns the three fitted rates, the observed spread, and the permutation p.
    """
    rng = np.random.default_rng(seed)
    matrices = [m[np.isfinite(m).any(axis=1)] for m in matrices]
    fitted = [Exp3.retention_rate(np.nanmean(m, axis=0)) for m in matrices]
    observed = np.ptp(fitted)
    pooled = np.vstack(matrices)
    sizes = [len(m) for m in matrices]

    extreme = 0
    for _ in range(n_perm):
        order = rng.permutation(len(pooled))
        start, spread = 0, []
        for size in sizes:
            spread.append(Exp3.retention_rate(np.nanmean(pooled[order[start:start + size]], axis=0)))
            start += size
        extreme += np.ptp(spread) >= observed
    return fitted, observed, (extreme + 1) / (n_perm + 1)


# ----------------------------------------------------------------- printing
def header(title):
    print(f"\n{title}\n{'-' * len(title)}")


def line(label, result):
    print(f"  {label:<44s}{result}")


def fmt_f(f, df1, df2, p, eta=None):
    degrees = f"{df2:g}" if float(df2).is_integer() else f"{df2:.1f}"
    text = f"F({df1},{degrees}) = {f:.3g}, p = {p:.3g}"
    return text + (f", np2 = {eta:.3f}" if eta is not None else "")


# ----------------------------------------------------------------- measures
def experiment_12():
    """One row per participant of Experiments 1-2, with the condition labels."""
    reaching = common.load("exp1_2")
    rows = []
    for condition, (response, prior, _) in Exp1_2.CONDITIONS.items():
        subset = reaching.loc[reaching["condition"] == condition]
        targets, participants, error = common.target_means(subset, "error_deg")
        slope = -common.slope_per_column(targets, error)

        target_matrix, _ = common.matrix(subset, "target_index")
        error_matrix, _ = common.matrix(subset, "error_deg")
        residual = common.detrend(target_matrix, error_matrix, Exp1_2.POLY_ORDER[condition])
        sd = np.sqrt(np.nanmean(residual ** 2, axis=0))

        _, _, rt = common.target_means(subset, "rt_ms")
        rt_mean = np.nanmean(common.moving_mean(rt), axis=0)
        # Quadratic coefficient of reaction time against target position, one per
        # participant.  A positive coefficient is the U-shape reported in the paper.
        curvature = np.full(len(participants), np.nan)
        for column in range(len(participants)):
            ok = np.isfinite(rt[:, column])
            if ok.sum() > 3:
                curvature[column] = np.polyfit(targets[ok], rt[ok, column], 2)[0]

        for i, participant in enumerate(participants):
            rows.append(dict(condition=condition, response=response, prior=prior,
                             participant=participant, slope=slope[i], sd=sd[i],
                             rt=rt_mean[i], curvature=curvature[i]))
    return pd.DataFrame(rows)


def se_index_table():
    """Per-participant sequential-effect index for each of the six conditions."""
    reaching = common.load("exp1_2")
    rows = []
    for condition, (response, prior, _) in Exp1_2.CONDITIONS.items():
        result = Exp1_2.serial_dependence(reaching, condition)
        subject = result["subject"] - np.nanmean(result["subject"])
        if prior == 60:
            index = subject[:, 8] - subject[:, 10]
        else:
            index = (np.nanmean(subject[:, [7, 8]], axis=1)
                     - np.nanmean(subject[:, [10, 11]], axis=1))
        for value in index:          # all participants; see Exp1_2.dependence_index
            rows.append(dict(condition=condition, response=response, prior=prior, index=value))
    table = pd.DataFrame(rows)
    return table.loc[np.isfinite(table["index"])]


def experiment_3():
    """Per-participant Experiment 3 measures: the index at every lag, and the 2-5-back index."""
    data = Exp3.load("exp3")
    out = {}
    for condition in Exp3.ORDER:
        rows = data.loc[data["condition"] == condition]
        target, participants = Exp3.matrix(rows, "target_deg")
        error, _ = Exp3.matrix(rows, "error_deg")
        setting = Exp3.SETTINGS[condition]

        deviation = Exp3.systematic_deviation(
            target, error, setting["poly_order"], setting["lookup_offset"])
        reference = Exp3.miniblock_reference(target) if setting["miniblock"] else target
        functions, index = Exp3.serial_dependence(deviation, reference, target)

        window = np.nanmean(functions[:, :, 1:5], axis=2)
        index_2to5 = Exp3.remove_outliers(window[:, 5] - window[:, 7], 2.5)
        out[condition] = dict(index=index, index_2to5=index_2to5, n=len(participants))
    return out


# ----------------------------------------------------------------- the tests
def report_experiment_12(table):
    header("Experiments 1-2: 2 (response contingency) x 3 (prior range), Type II sums of squares")
    names = ["prior range", "response contingency", "interaction"]
    for measure, label in [("slope", "bias slope"), ("sd", "motor variability")]:
        for (_, f, df1, df2, p, eta), name in zip(
                factorial_anova(table[measure].to_numpy(float), table["prior"], table["response"]), names):
            line(f"{label}, {name}", fmt_f(f, df1, df2, p, eta))

    header("Reaction time across prior ranges")
    for response, label in [("Free", "free response"), ("Delayed", "delayed response (Fig S2a)")]:
        subset = table.loc[table["response"] == response]
        f, df1, df2, p, eta = one_way_anova(
            [subset.loc[subset["prior"] == prior, "rt"] for prior in PRIORS])
        line(label, fmt_f(f, df1, df2, p, eta))

    header("U-shaped reaction time within condition (quadratic coefficient against zero)")
    for condition in FREE:
        values = table.loc[table["condition"] == condition, "curvature"].to_numpy(float)
        values = values[np.isfinite(values)]
        statistic, p = stats.ttest_1samp(values, 0)
        line(f"{condition}", f"t({len(values) - 1}) = {statistic:.2f}, p = {p:.3g}")

    header("Homogeneity of variance across the six groups (Levene, median-centred)")
    for measure, label in [("slope", "bias slope"), ("sd", "motor variability"),
                           ("rt", "reaction time")]:
        statistic, p = levene([table.loc[table["condition"] == c, measure] for c in Exp1_2.CONDITIONS])
        line(label, f"W = {statistic:.2f}, p = {p:.3g}" + ("  (variances differ)" if p < ALPHA else ""))


def report_sequential_effect():
    table = se_index_table()
    header(f"Sequential-effect index: 2 (response contingency) x 3 (prior range), N = {len(table)}")
    effects = factorial_anova(table["index"].to_numpy(float), table["prior"], table["response"])
    for (_, f, df1, df2, p, eta), name in zip(effects, ["prior range", "response contingency",
                                                        "interaction"]):
        line(name, fmt_f(f, df1, df2, p, eta))


def report_experiment_3():
    result = experiment_3()
    header("Experiment 3: the sequential effect across the three autocorrelation conditions")

    groups = [result[c]["index"][:, 0] for c in Exp3.ORDER]
    sizes = "/".join(str(int(np.isfinite(g).sum())) for g in groups)
    f, df1, df2, p = welch_anova(groups)
    line(f"1-back index, Welch (n = {sizes})", fmt_f(f, df1, df2, p))

    matrices = [result[c]["index"] for c in Exp3.ORDER]
    fitted, spread, p = permutation_retention(matrices)
    sizes = "/".join(str(int(np.isfinite(m).any(axis=1).sum())) for m in matrices)
    line(f"retention rate, permutation (n = {sizes})",
         f"rates {' / '.join(f'{v:.3f}' for v in fitted)}, spread = {spread:.3f}, p = {p:.4f}")

    groups = [result[c]["index_2to5"] for c in Exp3.ORDER]
    sizes = "/".join(str(int(np.isfinite(g).sum())) for g in groups)
    f, df1, df2, p = welch_anova(groups)
    line(f"2- to 5-back index, Welch (n = {sizes})", fmt_f(f, df1, df2, p))
    for a, b, q, df, p_adj in games_howell(groups, [Exp3.PSWITCH[c] for c in Exp3.ORDER]):
        mark = "*" if p_adj < ALPHA else " "
        print(f"      Games-Howell  P = {a} vs {b:<4s} q = {q:.2f}, df = {df:.1f}, p = {p_adj:.3g} {mark}")

    header("Experiment 3: how far back the sequential effect reaches (Fig 6d)")
    for condition in Exp3.ORDER:
        index = result[condition]["index"]
        marks, reach = [], 0
        for lag in range(index.shape[1]):
            values = index[np.isfinite(index[:, lag]), lag]
            statistic, p = stats.ttest_1samp(values, 0)
            repulsive = p < ALPHA and statistic > 0
            marks.append(f"{lag + 1}{'*' if repulsive else ' '}")
            if repulsive and reach == lag:
                reach = lag + 1
        print(f"    P(switch) = {Exp3.PSWITCH[condition]:<4s} {' '.join(marks)}   reaches {reach} trials")
    print("    * repulsive index above zero at p < 0.05")


def report_compensation():
    header("Figure 6c: how much of the attractive bias the sequential effect offsets")
    udl = Exp3.compensation_udl()
    random = Exp3.compensation_random()
    random = random[np.isfinite(random)]
    statistic, df, p, difference, half_width = Exp3.welch(udl, random)
    line(f"UDL against random targets (n = {len(udl)}/{len(random)})",
         f"t({df:.1f}) = {statistic:.2f}, p = {p:.3g}, np2 = {statistic ** 2 / (statistic ** 2 + df):.3f}")
    line("difference", f"{difference:+.1f}% [95% CI {difference - half_width:+.1f}, "
                       f"{difference + half_width:+.1f}]")


def report_counts():
    header("Participants")
    for name, label, order in [("exp1_2", "Experiments 1-2", list(Exp1_2.CONDITIONS)),
                               ("Simple_rt", "Experiment S1", ["s120", "s60"]),
                               ("exp3", "Experiment 3", Exp3.ORDER)]:
        table = common.load(name)
        people = table.groupby("condition")["participant"].nunique()
        counts = " + ".join(str(people[c]) for c in order)
        line(label, f"{people.sum()} participants ({counts})")


def main():
    report_experiment_12(experiment_12())
    report_sequential_effect()
    report_experiment_3()
    report_compensation()
    report_counts()


if __name__ == "__main__":
    # Empty delta-target bins are expected: the narrower prior ranges cannot fill every bin.
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom")
    main()
