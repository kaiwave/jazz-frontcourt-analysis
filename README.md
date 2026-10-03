# jazz-frontcourt-analysis
The Jazz Rim Protection and Frontcourt Pairing Model

An NBA defensive analytics framework combining Empirical Bayes modelling with a stochastic kinematic simulation to optimise weak-side help rotations across Utah Jazz frontcourt pairings. Built quickly and specifically to boost my application to the Basketball Analytics Intern position with the Utah Jazz.

## Implementation & Execution Checklist

### Phase 1: Environment & Repository Setup

* [x] Initialize git repo: `jazz-frontcourt-analysis`.

* [x] Set up `.gitignore`.

* [x] Create `requirements.txt` containing: `numpy`, `scipy`, `matplotlib`, `nba_api`, `pandas`, `shapely`.

* [x] Create directory placeholders: `assets/`, `data/`.

### Phase 2: Building `model.py`

* [x] Add court and biomechanical constants at the top of the script.

* [x] Implement `fetch_rim_defense_data()` with automatic CSV caching in `data/`.

* [x] Implement `fit_empirical_bayes_rim()` using Method of Moments and `scipy.stats.beta`.

* [x] Write `sample_drive_neighbourhood()` for bivariate normal perturbation around drive coordinates.

* [ ] Write `calc_help_rotation()` and vectorized `calc_rotation_risk()` for the kinematic race.

* [ ] Implement `solve_optimal_roam_depth()` using bisection over the $[R_{\text{ra}}, D_{\text{max}}]$ domain.

* [ ] Implement plotting functions with consistent styling, colourbars, and parameter labels.

### Phase 3: Writing & Executing `main.ipynb`

* [x] Configure notebook header with `%reload_ext autoreload` and `%autoreload 2`.

* [ ] Write Markdown sections for Introduction, Problem Formulation, and Mathematical Derivations.

* [ ] Run code cells to execute the model and output high-DPI assets into `assets/`.

* [ ] Write analytical interpretations beneath each generated graphic (explaining the kinematic curves and contour maps).

* [ ] Populate `references.bib` with citations (e.g., sports tracking kinematics, Bayesian modeling in basketball).

### Phase 4: Verification & Delivery

* [ ] Verify that all code in `model.py` runs without deprecation warnings (e.g., escape sequences in LaTeX labels).

* [ ] Verify that re-running all cells in `main.ipynb` executes cleanly from top to bottom.

* [ ] Confirm figures save properly in `assets/`.

* [ ] Test document rendering via `quarto render main.ipynb --to pdf`.