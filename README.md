# Utah Jazz Frontcourt Rim Protection and Help-Rotation Tethers
The Utah Jazz Rim Protection and Frontcourt Pairing Model

An NBA defensive analytics framework combining Empirical Bayes modeling with a stochastic kinematic simulation to optimize weak-side help rotations across Utah Jazz frontcourt pairings. Built quickly and specifically to boost an application to the Basketball Analytics Intern position with the Utah Jazz.

The compiled report can be found on my cloud [here](https://cloud.kaiwave.dev/s/ZjH9qaZKYAPYGWz)

## Intro
Open `main.ipynb` for the Executive Analytics Brief. It contains the Empirical Bayes rim baselines, roaming-depth heatmaps, and the frontcourt EV table. Rendered figures and the table are also saved in `assets/`.

## Headline results (2025-26, model outputs)
- **Drop anchor expands JJJ's safe roaming depth.** Relative to a switch, drop coverage adds about 1 to 3 ft of safe depth on average by region, up to 3.7 ft at the maximum (mean +2.7 ft for drivers 15 to 22 ft from the rim), worth about 3 points per 100 rim attempts from the wing and 8.8 from the short corner.

- **JJJ's rim suppression survives shrinkage.** Raw 53.7% on 270 attempts; Empirical Bayes posterior 55.0% (95% CI 49.4% to 60.6%).

- **Short-corner late rotations are the main cost.** P(late) passes 35% beyond about 16 ft (drop) or 14 ft (switch).

- **Directives.** With drop bigs, JJJ may shade to the nail (<= 20 ft) from the wing and should hold about 14 ft from the short corner. In switch lineups, use a strict 12 ft tether.

## Repository layout
| Path | Purpose |
|---|---|
| `main.ipynb` | Executive Analytics Brief (executed, with outputs) |
| `model.py` | Data ingestion, Empirical Bayes fit, kinematic simulation, plotting |
| `data/` | Cached NBA API rim defense tables (`< 6 ft`), 2022-23 through 2025-26 |
| `assets/` | Rendered Figure 1, Figure 2, and Table 1 (CSV) |

## Running it
```bash
pip install -r requirements.txt
jupyter lab main.ipynb
```
Run from the repository root. Data loads from `data/` when cached; otherwise it is fetched from the NBA API. The notebook fixes the random seed (7), so the Monte Carlo figures reproduce. The two court grids take roughly a minute and a half to compute.

## Method in brief
1. **Empirical Bayes.** A Beta prior is fit by method of moments on defenders with >= 10 games and >= 50 rim attempts (mean 65.0%, Beta(23.25, 12.54), about 36 pseudo-attempts). Each defender's posterior combines the prior with his makes and misses.
2. **Kinematics.** Drive time to the restricted area is compared with help time (0.22 s reaction + 18.5 ft/s sprint + 3.5 ft reach). Drives are sampled around a start point (sigma = 1.5 ft) to estimate P(late).
3. **Optimization.** Bisection finds d*, the deepest roaming position with P(late) below 5%.
4. **EV.** Expected FG% blends the roamer's and the anchor's posterior DFG% by P(late), converted to points per 100 rim attempts.

## Assumptions and limitations
- Constant speeds, 2D geometry, one help depth (20 ft) for the EV table.
- The perimeter defender's rim DFG% (64.5%) is a placeholder, not a measured value; EV gaps versus switching scale with it.
- Results are model outputs and have not been validated against lineup-level tracking or film.
- JJJ played 48 games, Nurkic 41, and Markkanen 42 in this data, so intervals are wide.
- The data lists Jaxson Hayes with LAL, so the notebook excludes him from the Utah highlight in Figure 1.

## Extra information (omitted from the brief for space; take or leave)
- **Sensitivity analysis.** Sweep driver speed, reaction time, and the 5% risk threshold, and show how d* and the EV gap move. This is the best way to back the size of the roaming gain.
- **Replace the perimeter placeholder** with a measured rim DFG% for actual switch-lineup defenders.
- **Pool seasons.** Use the 2022-23 to 2024-25 files for a multi-year prior or a hierarchical model with age effects.
- **Add a pairing library.** Run the same table for Hayes, Filipowski, and prospective trade targets as anchors.
- **Validate on film.** Tag a sample of short-corner drives and compare observed help arrival times with the model.
- **Speed up the grid.** Vectorize `calc_rotation_risk` over samples with NumPy to make finer grids practical.
- **Add tests** for the Beta posterior and the bisection solver.
