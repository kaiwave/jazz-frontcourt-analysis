import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats
from nba_api.stats.endpoints import leaguedashptdefend
import pandas as pd

# ---------------------------------------
# CONSTANTS & PARAMETERS
# ---------------------------------------
# Court dimensions
R_RIM:             float = 0.75     # ft
R_RESTRICTED_AREA: float = 4.0      # ft
KEY_W:             float = 16.0     # ft
KEY_L:             float = 19.0     # ft

# Kinematic parameters
V_DRIVE_D:         float = 14.5     # ft/s
V_DRIVE_ISO:       float = 17.5     # ft/s
V_SPRINT:          float = 18.5     # ft/s
T_REACT:           float = 0.22     # s
L_REACH:           float = 3.5      #ft

# Filter parameters
MIN_GAMES:         int = 10         # Minimum number of games to include a player
FGA_LT_06FT:       int = 50         # Minimum number of FGA < 6ft to include a player

# Stochastic model parameters
SIGMA_X_DEFAULT: float = 1.5        # ft, chosen since NBA step size is ~1.5 ft
SIGMA_Y_DEFAULT: float = 1.5        # ft, chosen since NBA step size is ~1.5 ft
RISK_THRESHOLD:  float = 0.05       # Threshold for risk of late help rotation

# Directories
DATA_DIR:          str = "data"     # Directory to store cached data
UTA_ID:            int = 1610612747 # Utah Jazz team ID in NBA API

# ---------------------------------------
# DATA INGESTION
# ---------------------------------------
def fetch_rim_defense_data(
  season: str, 
  force_refresh: bool = False
  ) -> pd.DataFrame:

  os.makedirs(DATA_DIR, exist_ok=True)
  file_path = os.path.join(DATA_DIR, f"rim_defense_{season}.csv")

  if not os.path.exists(file_path) or force_refresh:
    try:
      print(f"Downloading rim defense data for {season} from NBA API...")

      response = leaguedashptdefend.LeagueDashPtDefend(
        defense_category = 'Less Than 6Ft',
        season = season,
        season_type_all_star = 'Regular Season',
        per_mode_simple= 'Totals',
        timeout = 30
      )

      df = response.get_data_frames()[0] # Get the main leaderboard table from the response

      print("Saving to:", os.path.abspath(file_path))
      df.to_csv(file_path, index=False) # Save to cache

      return df

    except Exception as e:
      print(f"Failed to download rim defense data for {season}: {e}")

  else:
    print(f"Loading rim defense data for {season} from cache...")

    df = pd.read_csv(file_path)
    return df # Return the cached DataFrame

  return pd.DataFrame() # In case of failure, return an empty DataFrame

# ---------------------------------------
# STATISTICAL AND STOCHASTIC MODEL
# ---------------------------------------
def fit_empirical_bayes_rim(
    df: pd.DataFrame,
    min_games: int = MIN_GAMES,
    fga_lt_06ft: int = FGA_LT_06FT
    ) -> tuple[pd.DataFrame, float, float, float, float]:

  filtered_df = df[(df['GP'] >= min_games) & (df['FGA_LT_06'] >= fga_lt_06ft)] # Filter players with sufficient games and FGA < 6ft for baseline calcs

  p_mean = filtered_df['LT_06_PCT'].mean()
  p_var = filtered_df['LT_06_PCT'].var()

  kappa = (p_mean * (1.0 - p_mean) / p_var) - 1.0

  alpha_0 = p_mean * kappa
  beta_0 = (1.0 - p_mean) * kappa # Hyperparameters for the Beta prior distribution
  
  out_df = df.copy() # Copy the full dataframe

  out_df['alpha_post'] = out_df['FGM_LT_06'] + alpha_0
  out_df['beta_post'] = (out_df['FGA_LT_06'] - out_df['FGM_LT_06']) + beta_0   # Posterior shape parameters

  out_df['post_dfg_pct'] = out_df['alpha_post'] / (out_df['alpha_post'] + out_df['beta_post'])  # Posterior Mean

  out_df['post_dfg_pct_lower'] = stats.beta.ppf(0.025, out_df['alpha_post'], out_df['beta_post'])  # 2.5th percentile of the posterior distribution
  out_df['post_dfg_pct_upper'] = stats.beta.ppf(0.975, out_df['alpha_post'], out_df['beta_post'])  # 97.5th percentile of the posterior distribution

  out_df['eb_plusminus'] = out_df['post_dfg_pct'] - out_df['NS_LT_06_PCT'] # Empirical Bayes Plus-Minus
  out_df['pts_saved_per_100'] = -1 * out_df['eb_plusminus'] * 100 * 2.0 # Points saved per 100 rim attempts, -ve because a lower DFG% is better

  return out_df, alpha_0, beta_0, p_mean, kappa


def sample_drive_neighbourhood(
    p0: tuple[float, float],
    sigma_x: float = SIGMA_X_DEFAULT,
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = 1000
    ) -> np.ndarray:

  sample_matrix = np.random.normal(loc=p0, scale=[sigma_x, sigma_y], size=(n_samples, 2))

  return sample_matrix

# ---------------------------------------
# OPTIMISATION
# ---------------------------------------
def calc_help_rotation(
    p0: tuple[float, float],
    d_help: float,
    v_drive: float = V_DRIVE_D,
    ) -> tuple[float, float, float, float]:

  dist_drive = max(0.01, np.hypot(p0[0], p0[1]) - R_RESTRICTED_AREA) # Distance from rim to defender, clamped to a minimum of 0.1 ft to avoid division by zero

  t_drive = dist_drive / v_drive # Time for the drive to reach the rim

  delta_s = max(0.01, d_help - R_RESTRICTED_AREA - L_REACH) # Distance the help defender needs to cover, clamped to a minimum of 0.1 ft to avoid division by zero

  t_help = T_REACT + (delta_s / V_SPRINT) # Time for the help defender to reach the rim

  delta_t = t_help - t_drive # Time difference between the help defender and the drive

  return t_drive, t_help, delta_t, dist_drive

def calc_rotation_risk(
    p0: tuple[float, float],
    d_help: float,
    v_drive: float = V_DRIVE_D,
    samples: np.ndarray | None = None
) -> float:

  if samples is None:
    samples = sample_drive_neighbourhood(p0) # Sample the drive neighbourhood around the defender's position

  if len(samples) == 0:
    return 0.0 # No samples -> no risk

  late_count = 0
  for sample in samples:
    _, _, delta_t, _ = calc_help_rotation((sample[0], sample[1]), d_help, v_drive)
    if delta_t > 0.0:
      late_count += 1

  risk = float(np.clip((late_count / len(samples)), 0.0, 1.0))

  return risk

def solve_optimal_roam_depth(
    p0: tuple[float, float],
    v_drive: float = V_DRIVE_D,
    risk_threshold: float = RISK_THRESHOLD,
    n_steps: int = 40,
    ):

  samples = sample_drive_neighbourhood(p0)

  lower_0 = R_RESTRICTED_AREA + L_REACH
  upper_0 = 23.75 # ft, 3pt line distance from the rim

  for _ in range(n_steps):
    mid_0 = (lower_0 + upper_0) / 2.0
    risk = calc_rotation_risk(p0, mid_0, v_drive, samples)

    if risk < risk_threshold:
      lower_0 = mid_0 # Increase depth to reduce risk
    else:
      upper_0 = mid_0 # Decrease depth to increase defensive coverage

  return lower_0

# ---------------------------------------
# SPECIFIC COMPUTATIONS
# ---------------------------------------
def simulate_court_grid(
    grid_resolution: int = 20,
    v_drive: float = V_DRIVE_D,
    risk_threshold: float = RISK_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

  x_coords = np.linspace(-25.0, 25.0, grid_resolution)
  y_coords = np.linspace(0.0, 41.75, grid_resolution) # Establish court boundaries

  d_optimal_grid = np.zeros((grid_resolution, grid_resolution)) # Initialise a grid to store optimal roam depths

  for i, y_coord in enumerate(y_coords):
    for j, x_coord in enumerate(x_coords): # Since its stored as [y,x] in the grid, we need to iterate over y first and then x
      p0 = (x_coord, y_coord)
      d_optimal_grid[i,j] = solve_optimal_roam_depth(p0, v_drive, risk_threshold) # Store the optimal roam depth for each grid point

  return x_coords, y_coords, d_optimal_grid

def calc_pairing_ev(
    risk: float,
    roam_dfg: float,
    anchor_dfg: float,
    ) -> tuple[float, float, float]:

  exp_fg_pct = (1-risk) * (roam_dfg) + risk * (anchor_dfg) # Expected field goal percentage based on the risk of late help rotation

  exp_pts_per_100 = exp_fg_pct * 2.0 * 100.0 # Expected points per 100 rim attempts based on the expected field goal percentage

  pts_saved_per_100 = (anchor_dfg - exp_fg_pct) * 2.0 * 100.0 # Expected points saved per 100 rim attempts 

  return exp_fg_pct, exp_pts_per_100, pts_saved_per_100

def eval_jazz_frontcourt_pairing( # Jazz specific function to evaluate the expected value of a frontcourt pairing. 
    roam_dfg: float = 0.551,   # JJJ's DFG% < 6ft from 2025-26 season
    anchor_dfg: float = 0.600, # Nurkic's DFG% < 6ft from 2025-26 season
    eb_df: pd.DataFrame | None = None,
    d_help: float = 20.0
    ) -> pd.DataFrame:

  if eb_df is not None:
    jjj_row = eb_df[eb_df['CLOSE_DEF_PERSON_ID'] == 1628991]
    nurk_row = eb_df[eb_df['CLOSE_DEF_PERSON_ID'] == 203994]
    if not jjj_row.empty:
      roam_dfg = float(jjj_row['post_dfg_pct'].iloc[0])
    if not nurk_row.empty:
      anchor_dfg = float(nurk_row['post_dfg_pct'].iloc[0]) # Pull the empirical Bayes DFG% for JJJ and Nurkic if available

  # Model testing spots
  TEST_SPOTS = {
    "Top of Key (3pt)": (0.0, 25.0),
    "Wing / Elbow": (15.0, 15.0),
    "Short Corner / Slot": (12.0, 8.0),
  } # Define some test spots on the court to evaluate the pairing

  SCHEMES = {
    "Drop Anchor" : {
      "v_drive": V_DRIVE_D,
      "anchor_dfg": anchor_dfg,
    },

    "Switch/Perimeter" : {
      "v_drive": V_DRIVE_ISO,
      "anchor_dfg": 0.645, # Estimated DFG% < 6ft for a perimeter defender, based on league average
    }
  }

  rows = []

  for spot_name, p0 in TEST_SPOTS.items():
    for scheme_name, scheme_params in SCHEMES.items():
      scheme_v_drive = scheme_params["v_drive"]
      scheme_anchor_dfg = scheme_params["anchor_dfg"]

      solve_depth = solve_optimal_roam_depth(p0, scheme_v_drive, RISK_THRESHOLD) # Solve for the optimal roam depth for the given spot and scheme

      risk = calc_rotation_risk(p0, d_help, scheme_v_drive) # Calculate the risk of late help rotation for the given spot and scheme

      exp_values = calc_pairing_ev(risk, roam_dfg, scheme_anchor_dfg) # Calculate the expected values of the stuff for the given spot and scheme

      rows.append({
        "Spot": spot_name,
        "Scheme": scheme_name,
        "Roam DFG%": roam_dfg,
        "Anchor DFG%": scheme_anchor_dfg,
        "Optimal Roam Depth (ft)": solve_depth,
        "Risk of Late Help Rotation": risk,
        "Expected FG%": exp_values[0],
        "Expected Points per 100 Rim Attempts": exp_values[1],
        "Expected Points Saved per 100 Rim Attempts": exp_values[2],
      }) # Append the results to the rows list

  return pd.DataFrame(rows) # Return the results as a DataFrame

# ---------------------------------------
# VISUALISATION
# ---------------------------------------
def draw_nba_halfcourt():
  pass

def plot_empirical_bayes_forest():
  pass

def plot_help_window():
  pass

def plot_recovery_risk_curve():
  pass

def plot_zone_roam_surface():
  pass