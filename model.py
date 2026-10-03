import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
def calc_help_rotation():
  pass

def calc_rotation_risk():
  pass

def solve_optimal_roam_depth():
  pass

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