import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import minimize_scalar
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

# Directories
DATA_DIR:          str = "data"  # Directory to store cached data
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
def fit_empirical_bayes_rim():
  pass

def sample_drive_neighbourhood():
  pass

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