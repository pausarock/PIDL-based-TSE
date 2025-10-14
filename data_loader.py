import os
import numpy as np
import pandas as pd

DATA_DEFAULTS = {
    "flow_csv": "sh_flow_data.csv",
    "speed_csv": "sh_speed_data.csv",
}


def load_flow_speed(data_dir: str = ".", flow_csv: str | None = None, speed_csv: str | None = None):
    flow_csv = flow_csv or DATA_DEFAULTS["flow_csv"]
    speed_csv = speed_csv or DATA_DEFAULTS["speed_csv"]
    flow_path = os.path.join(data_dir, flow_csv)
    speed_path = os.path.join(data_dir, speed_csv)

    flow = np.array(pd.read_csv(flow_path).iloc[:, 1:].values, dtype=float)
    speed = np.array(pd.read_csv(speed_path).iloc[:, 1:].values, dtype=float)

    t_count, x_count = flow.shape
    assert speed.shape == (t_count, x_count)

    return flow, speed


def build_mesh(x: np.ndarray, t: np.ndarray):
    X, T = np.meshgrid(x.astype(float), t.astype(float))
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    return X, T, X_star


def metrics_mape(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true != 0
    return np.fabs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()


def metrics_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mse = np.mean((y_true.flatten() - y_pred.flatten()) ** 2)
    return float(np.sqrt(mse))
