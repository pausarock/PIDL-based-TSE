import numpy as np
import pandas as pd
from typing import Tuple, List

def load_topology_data():
    """Load topology data from CSV files"""
    import os
    # Get the parent directory where CSV files are located
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load detector positions
    detector_df = pd.read_csv(os.path.join(parent_dir, 'detector_positions.csv'))
    X_DETECTORS = detector_df['position_km'].values.astype(float)
    
    # Load ramp information
    ramp_df = pd.read_csv(os.path.join(parent_dir, 'ramp_info.csv'))
    RAMP_LOC = ramp_df['position_km'].tolist()
    RAMP_TYPE = ramp_df['ramp_type'].tolist()
    
    # Load station configuration
    station_df = pd.read_csv(os.path.join(parent_dir, 'station_config.csv'))
    USE_ID = station_df[station_df['station_type'] == 'supervised']['station_id'].tolist()
    UNCOVER_ID = station_df[station_df['station_type'] == 'hidden']['station_id'].tolist()
    
    return X_DETECTORS, RAMP_LOC, RAMP_TYPE, USE_ID, UNCOVER_ID

# Load topology data
X_DETECTORS, RAMP_LOC, RAMP_TYPE, USE_ID, UNCOVER_ID = load_topology_data()


def region_N(x_grid: np.ndarray) -> np.ndarray:
    """Compute N(x) piecewise region indicator over grid X.
    N=3 in specified segments; otherwise 2.
    Accepts 2D grid (T, X) or 1D positions.
    """
    arr = np.zeros_like(x_grid, dtype=float)
    arr[(x_grid>=0.5)&(x_grid<3.2)] = 3
    arr[(x_grid>=5.6)&(x_grid<7.45)] = 3
    arr[(x_grid>=8.1)&(x_grid<8.5)] = 3
    arr[(x_grid>=12.05)&(x_grid<12.6)] = 3
    arr[(arr==0)] = 2
    return arr


def get_topology() -> Tuple[np.ndarray, List[float], List[int], list, list]:
    return X_DETECTORS.copy(), RAMP_LOC.copy(), RAMP_TYPE.copy(), USE_ID.copy(), UNCOVER_ID.copy()
