import numpy as np
from typing import Tuple, List

# Detector longitudinal positions along the corridor (km)
X_DETECTORS = np.array([0,0.38,1.23,2.07,2.58,2.99,3.43,3.88,4.66,5.07,5.42,5.72,
             6.47,7.24,7.59,7.88,8.33,8.77,9.20,9.59,10.24,10.79,
             11.96,12.30,12.68,13.53,13.95,15.26,16.17,16.64,17.06,17.47], dtype=float)

# Ramp locations (km) and types (+1 on-ramp, -1 off-ramp)
RAMP_LOC = [0.65,1.8,3.2,4.2,4.9,5.65,7.12,7.42,8.15,8.97,9.39,10.5,11.5,12.07,12.29,12.6,14.15,15.5,16.5,17.2]
RAMP_TYPE = [1,-1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,1]

# Indices used for supervised boundary stations
USE_ID = [0,2,4,7,8,10,11,13,14,16,17,18,20,21,22,23,24,26,27,28,30,31]
# Hidden stations for test visualization (subset)
UNCOVER_ID = [1,3,5,6,9,12,15,19,25,29]


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
