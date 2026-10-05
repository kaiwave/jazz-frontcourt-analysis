import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from nba_api.stats.endpoints import leaguedashptdefend

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

# IDs for Jazz frontcourt players
JAZZ_FRONTCOURT_IDS = [
    1628991,  # Jaren Jackson Jr.
    203994,   # Jusuf Nurkić
    1629637,  # Jaxson Hayes
    1628374,  # Lauri Markkanen
    1642271,  # Kyle Filipowski
]

# Custom colormap for Jazz-themed visualisations
jazz_colors = ['#31006F', '#00A9E0', "#062553"]
jazz_cmap = ListedColormap(jazz_colors, name='utah_jazz')

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
      late_count += 1 # Count the number of samples where the help defender is late to the rim

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
def draw_nba_halfcourt(
    ax: Axes | None = None,
    color: str = "black",
    lw: float = 1.5,
    zorder: int = 10,
) -> Axes:

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7.5))

    ax.plot([-3.0, 3.0], [-1.25, -1.25], color=color, lw=lw * 1.5, zorder=zorder)                                          # Backboard: 6 ft wide, 4 ft inside baseline (-5.25 + 4.0 = -1.25)

    ax.plot([-22.0, -22.0], [-5.25, 8.75], color=color, lw=lw, zorder=zorder)
    ax.plot([22.0, 22.0], [-5.25, 8.75], color=color, lw=lw, zorder=zorder)                                                # Corner 3-point lines: 22 ft from center, running 14 ft from baseline (-5.25 to 8.75)

    court_patches = [
        patches.Rectangle((-25.0, -5.25), 50.0, 47.0, fill=False, edgecolor=color, lw=lw, zorder=zorder),                  # Outer half-court boundary (50 ft wide x 47 ft long)
        patches.Rectangle((-8.0, -5.25), 16.0, 19.0, fill=False, edgecolor=color, lw=lw, zorder=zorder),                   # Outer Key (16 ft wide x 19 ft long from baseline)
        patches.Rectangle((-6.0, -5.25), 12.0, 19.0, fill=False, edgecolor=color, lw=lw * 0.75, zorder=zorder),            # Inner Key (12 ft wide x 19 ft long)
        patches.Circle((0.0, 0.0), radius=0.75, fill=False, edgecolor=color, lw=lw, zorder=zorder),                        # Rim (18-inch diameter = 0.75 ft radius)
        patches.Arc((0.0, 0.0), 8.0, 8.0, theta1=0.0, theta2=180.0, edgecolor=color, lw=lw, zorder=zorder),                # Restricted Area Arc (4 ft radius from center of rim)
        patches.Arc((0.0, 13.75), 12.0, 12.0, theta1=0.0, theta2=180.0, edgecolor=color, lw=lw, zorder=zorder),            # Free-Throw Circle - Top Half (solid, centered at y = 13.75, 6 ft radius)
        patches.Arc((0.0, 13.75), 12.0, 12.0, theta1=180.0, theta2=360.0, edgecolor=color, lw=lw, ls="--", zorder=zorder), # Free-Throw Circle - Bottom Half (dashed)
        patches.Arc((0.0, 0.0), 47.5, 47.5, theta1=22.0, theta2=158.0, edgecolor=color, lw=lw, zorder=zorder),             # 3-Point Arc (23.75 ft radius, connecting the corner 3 lines from 22 deg to 158 deg)
        patches.Arc((0.0, 41.75), 12.0, 12.0, theta1=180.0, theta2=360.0, edgecolor=color, lw=lw, zorder=zorder),          # Half-Court Center Circle (centered at y = 41.75, 6 ft radius)
    ]

    for patch in court_patches:
        ax.add_patch(patch)

    ax.set_aspect("equal")     # Lock aspect ratio and coordinate bounds
    ax.set_xlim(-26.0, 26.0)
    ax.set_ylim(-6.0, 43.0)
    ax.set_xlabel("Court Width (ft from Rim Center)")
    ax.set_ylabel("Court Depth (ft from Rim Center)")

    return ax

def plot_empirical_bayes_forest(
    eb_df: pd.DataFrame,
    top_n: int = 12,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5)) # Create a new figure and axes if none are provided

    jazz_df = eb_df[eb_df["CLOSE_DEF_PERSON_ID"].isin(JAZZ_FRONTCOURT_IDS)] 
    top_df = eb_df.nsmallest(top_n, "post_dfg_pct") # Select the best defenders
    
    sub_df = (
        pd.concat([jazz_df, top_df])
        .drop_duplicates(subset=["CLOSE_DEF_PERSON_ID"])
        .sort_values("post_dfg_pct", ascending=False)  # Best (lowest) at the top of y-axis
        .reset_index(drop=True)
    )

    y_pos = np.arange(len(sub_df))
    is_jazz = sub_df["CLOSE_DEF_PERSON_ID"].isin(JAZZ_FRONTCOURT_IDS) # Create a boolean mask to identify Jazz players in the subset

    prior_mean = eb_df["LT_06_PCT"].mean()
    ax.axvline(
        prior_mean,
        color="crimson",
        linestyle="--",
        lw=1.5,
        label=f"League Mean ({prior_mean:.1%})", 
    ) # Draw a vertical line for the league mean DFG% < 6ft

    ax.hlines(
        y=y_pos[~is_jazz],
        xmin=sub_df.loc[~is_jazz, "post_dfg_pct_lower"],
        xmax=sub_df.loc[~is_jazz, "post_dfg_pct_upper"],
        colors="steelblue",
        alpha=0.45,
        lw=2.5,
        label="NBA Benchmarks (95% CI)",
    ) # Non Jazz players' 95% confidence intervals for their posterior DFG% < 6ft

    ax.hlines(
        y=y_pos[is_jazz],
        xmin=sub_df.loc[is_jazz, "post_dfg_pct_lower"],
        xmax=sub_df.loc[is_jazz, "post_dfg_pct_upper"],
        colors="#4B0082",
        alpha=0.85,
        lw=3.0,
        label="Utah Frontcourt (95% CI)",
    ) # Jazz players' 95% confidence intervals for their posterior DFG% < 6ft

    ax.scatter(
        sub_df["LT_06_PCT"],
        y_pos,
        color="white",
        edgecolor="dimgray",
        s=55,
        zorder=3,
        label="Raw Rim DFG% (LT_06_PCT)",
    ) # Scatter plot of the raw rim DFG% < 6ft for all players in the subset
    
    ax.scatter(
        sub_df.loc[~is_jazz, "post_dfg_pct"],
        y_pos[~is_jazz],
        color="steelblue",
        s=65,
        zorder=4,
        label="Posterior DFG% (NBA Top 12)",
    ) # Scatter plot of the posterior DFG% < 6ft for non Jazz players in the subset

    ax.scatter(
        sub_df.loc[is_jazz, "post_dfg_pct"],
        y_pos[is_jazz],
        color="#4B0082",
        s=80,
        zorder=5,
        label="Posterior DFG% (Utah Frontcourt)",
    ) # Scatter plot of the posterior DFG% < 6ft for Jazz players in the subset

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_df["PLAYER_NAME"])
    for tick_label, jazz_flag in zip(ax.get_yticklabels(), is_jazz):
        if jazz_flag:
            tick_label.set_fontweight("bold")
            tick_label.set_color("#4B0082") # Bold Utah player names on the y-axis

    ax.set_xlabel("Opponent Rim FG% Allowed (< 6 ft)")
    ax.set_title("Empirical Bayes Rim Protection: Utah Frontcourt vs. Top 12 NBA")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    return ax

def plot_zone_roam_surface(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    d_optimal_grid: np.ndarray,
    scheme_label: str = "Drop Anchor (v = 14.5 ft/s)",
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 7.5))

    draw_nba_halfcourt(ax=ax, color="black", lw=1.5, zorder=10) # Draw the NBA half-court on the axes

    levels = np.linspace(7.5, 23.75, 14)
    cf = ax.contourf(
        x_coords,
        y_coords,
        d_optimal_grid,
        levels=levels,
        cmap="jazz_cmap",
        alpha=0.85,
        extend="both",
        zorder=1,
    ) # Create a filled contour plot of the optimal roam depth surface

    line_levels = [9.0, 12.0, 15.0, 18.0, 21.0]
    cs = ax.contour(
        x_coords,
        y_coords,
        d_optimal_grid,
        levels=line_levels,
        colors="black",
        linewidths=0.9,
        linestyles="--",
        alpha=0.7,
        zorder=5,
    ) # Create contour lines for specific optimal roam depth levels
    ax.clabel(cs, inline=True, fontsize=9, fmt="%.0f ft")

    # 4. Colorbar & Titles
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Max Safe Roaming Depth d* (ft from Rim)", fontsize=10)

    ax.set_title(f"JJJ Spatial Roaming Tether Surface — {scheme_label}", fontsize=12, fontweight="bold")

    return ax