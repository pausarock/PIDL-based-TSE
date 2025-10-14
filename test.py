import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import torch

from .data_loader import load_flow_speed, metrics_mape, metrics_rmse
from .topology_loader import get_topology, region_N
from .pinn import PINN_JWZ, get_device


def main_test(data_dir: str = ".", weights_path: str = "best_model.pat", layers=None):
    device = get_device()
    x, ramp_loc, ramp_type, use_id, uncover_id = get_topology()
    flow, speed = load_flow_speed(data_dir)

    t = np.linspace(0.0, 24 - 24 / 288, 288).astype(float)

    # Build model
    X_star_bounds = np.array([[x.min(), t.min()], [x.max(), t.max()]], dtype=float)
    lb = X_star_bounds.min(0)
    ub = X_star_bounds.max(0)
    layers = layers or [2, 128, 128, 128, 64, 2]
    model = PINN_JWZ(layers, lb, ub, ramp_loc=ramp_loc, ramp_type=ramp_type, device=device)
    model.to(device)

    # Resolve weights path and load
    if not os.path.isabs(weights_path):
        candidate = os.path.join(os.getcwd(), weights_path)
        if os.path.exists(candidate):
            weights_path = candidate
        else:
            pkg_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", weights_path)
            if os.path.exists(pkg_candidate):
                weights_path = pkg_candidate

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}. Provide --weights_path to a valid .pat file, or run training to create it.")

    # sanity check: very small files are often corrupted/incomplete
    try:
        if os.path.getsize(weights_path) < 1024:
            raise RuntimeError("checkpoint file is too small (<1KB)")
    except OSError:
        pass

    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load weights from {weights_path}. Ensure it was saved with torch.save(model.state_dict(), path) "
            f"and not an empty/partial file. You can re-run training to regenerate it. Original error: {e}"
        )

    model.eval()

    # Predict on hidden detectors
    uncover = np.array(x[uncover_id]).astype(float)
    X_p, T_p = np.meshgrid(uncover, t)
    N = region_N(X_p)
    X_star_p = np.hstack((X_p.flatten()[:, None], T_p.flatten()[:, None]))

    start_time = time.time()
    u_pred, f_pred = model.predict(X_star_p)
    elapsed = time.time() - start_time

    print(f"预测计算时间: {elapsed:.4f} 秒")
    print(f"预测样本数: {X_star_p.shape[0]}")
    print(f"平均每样本时间: {elapsed / X_star_p.shape[0] * 1000:.4f} 毫秒")

    # Ground truth
    u_star_pred = np.hstack((flow[:, uncover_id].flatten()[:, None], speed[:, uncover_id].flatten()[:, None]))

    mape_u = metrics_mape(u_pred, u_star_pred[:, 1:2])
    mape_f = metrics_mape(f_pred, flow[:, uncover_id].flatten()[:, None])
    rmse_u = metrics_rmse(u_pred, u_star_pred[:, 1:2])
    rmse_f = metrics_rmse(f_pred, flow[:, uncover_id].flatten()[:, None])

    print(mape_u, mape_f, rmse_u, rmse_f / 60.0)

    # Save CSVs
    df_flow = pd.DataFrame(f_pred.reshape(288, -1))
    df_speed = pd.DataFrame(u_pred.reshape(288, -1))
    df_flow.to_csv(os.path.join(data_dir, "pred_flow_PW.csv"), index=False)
    df_speed.to_csv(os.path.join(data_dir, "pred_speed_PW.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--weights_path", type=str, default="best_model.pat")
    parser.add_argument("--layers", type=str, default="2,128,128,128,64,2")
    args = parser.parse_args()

    data_root = os.path.dirname(os.path.abspath(__file__))
    resolved_data_dir = args.data_dir if args.data_dir not in (None, "") else os.path.join(data_root, "..")
    layers_parsed = [int(x) for x in args.layers.split(',') if x.strip() != ""]

    main_test(data_dir=resolved_data_dir, weights_path=args.weights_path, layers=layers_parsed)
