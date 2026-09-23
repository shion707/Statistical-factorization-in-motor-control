# Statistical factorization in motor control

Code and data to reproduce the figures in Wang & Whitney, *Statistical factorization in motor control*.

## Getting started

You need Python 3.9 or newer with four common packages:

```bash
pip install numpy pandas scipy matplotlib
```

Tested with Python 3.11, numpy 2.1, pandas 2.2, scipy 1.14 and matplotlib 3.9.

Run each script from this folder, for example `python Exp1_2.py`.  Each one finishes in
about ten seconds or less.  The figure scripts open their figure in a window, and all but
`SE_simulation.py` print the numbers behind it; `stats.py` prints the tests.  To write the
figures to PNG and PDF files as well, which you need on a machine with no display, set
`SAVE_FIGURES = True` in `common.py`.

## What each script produces

| Script | Figures | In one line |
|---|---|---|
| `Exp1_2.py` | 1e, 1g, 3b–g, 5g | Reaching bias, reaction time and serial dependence across six prior conditions |
| `Exp3.py` | 6c–h | How serial dependence persists when the target sequence is autocorrelated |
| `SDDM.py` | 3 (model) | A population drift-diffusion model drawn over the Experiment 2 data |
| `SDDM_udl.py` | 4a–b | Bimodal reaching error after use-dependent learning, data beside model |
| `SE_simulation.py` | 5a–f | Why a repulsive serial dependence reduces total error |
| `stats.py` | — | The statistical tests reported in the paper |

`common.py` holds the helpers the scripts share.

## The data

| Table | Rows | Contents |
|---|---|---|
| `data/exp1_2.csv` | 141,120 | Experiments 1–2. Six conditions: `d180 d120 d60` delayed, `f180 f120 f60` free. 196 participants × 720 trials |
| `data/Simple_rt.csv` | 41,760 | Simple-reaction-time control, conditions `s120` and `s60`. 58 participants |
| `data/exp3.csv` | 170,600 | Experiment 3. Conditions `random`, `medium`, `high` by target autocorrelation. 36 / 40 / 49 participants |
| `data/udl_reaching.csv` | 37,656 | Tsay et al. (2022) use-dependent learning, experiments `E1` and `E2`, aligned to the trained direction |
