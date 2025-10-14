import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from scipy.interpolate import griddata
from pyDOE import lhs

from .data_loader import load_flow_speed, metrics_mape, metrics_rmse
from .topology_loader import get_topology, region_N
from .pinn import PINN_JWZ, get_device, estimate_flops_single_forward, count_parameters

alpha1 = 1/400
alpha2 = 60
beta1 = 1/1000
beta2 = 1/3000
gamma = 1/2000


def parse_int_list(csv: str):
    return [int(x.strip()) for x in csv.split(',') if x.strip() != '']


def parse_int_list_default(csv: str | None, default_list: list[int]) -> list[int]:
    if csv is None or str(csv).strip() == "":
        return list(default_list)
    return parse_int_list(csv)


def parse_int_list_layers(csv: str | None, default_list: list[int]) -> list[int]:
    if csv is None or str(csv).strip() == "":
        return list(default_list)
    return [int(x.strip()) for x in csv.split(',') if x.strip() != '']


def generate_S(exact_flow: np.ndarray, detect_id: list[int]) -> np.ndarray:
    S = exact_flow[:, detect_id[1]] - exact_flow[:, detect_id[0]]
    for i in range(len(detect_id) - 2):
        r_out = exact_flow[:, detect_id[i + 2]] - exact_flow[:, detect_id[i + 1]]
        S = np.vstack((S, r_out))
    return S.T



def atomic_save_state_dict(model: torch.nn.Module, path: str):
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    base_name = os.path.basename(path)
    tmp_path = os.path.join(dir_name, base_name + f".tmp.{int(time.time()*1000)}")
    torch.save(model.state_dict(), tmp_path)

    last_err = None
    for _ in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as e:
            last_err = e
            try:
                if os.path.exists(path):
                    os.remove(path)
                os.replace(tmp_path, path)
                return
            except Exception as e2:
                last_err = e2
                time.sleep(0.2)
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    finally:
        if last_err is not None:
            raise last_err



def main_train(
    data_dir: str = ".",
    total_steps: int = 30000,
    save_path: str = "best_model.pat",
    layers: list[int] | None = None,
    id_list: list[int] | None = None,
    s_detect_id: list[int] | None = None,
    alpha1: float = 1/400,
    alpha2: float = 60.0,
    beta1: float = 1/1000,
    beta2: float = 1/3000,
    gamma: float = 1/2000,
):
    device = get_device()

    default_layers = [2, 128, 128, 128, 64, 2]
    layers = layers or default_layers
    default_id_list = [2,4,7,8,10,11,13,14,16,17,18,20,21,22,23,24,26,27,28,30,31]
    id_list = id_list or default_id_list
    default_s_detect_id = [0,2,4,7,8,10,11,13,14,16,18,20,21,22,23,24,26,27,28,30,31]
    s_detect_id = s_detect_id or default_s_detect_id

    utn = 288
    tlo = 0.0
    thi = 24 - 24 / 288

    x, ramp_loc, ramp_type, use_id, uncover_id = get_topology()
    flow, speed = load_flow_speed(data_dir)

    t = np.linspace(tlo, thi, utn).astype(float)
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = np.hstack((flow.flatten()[:, None], speed.flatten()[:, None]))

    N = region_N(X)
    Z_flow = flow[:, use_id] / 6000.0

    X_u_train = np.hstack((X[:, 0:1], T[:, 0:1], N[:, 0:1]))
    u_train = np.hstack((flow[:, 0:1], speed[:, 0:1]))

    for i, idd in enumerate(id_list):
        xx = np.hstack((X[:, idd:idd+1], T[:, idd:idd+1], N[:, idd:idd+1]))
        uu = np.hstack((flow[:, idd:idd+1], speed[:, idd:idd+1]))
        X_u_train = np.vstack([X_u_train, xx])
        u_train = np.vstack([u_train, uu])

    lb = X_star.min(0)
    ub = X_star.max(0)

    sample_x = lhs(1, 500) * 0.5 + ramp_loc[0] - 0.25
    for i in range(19):
        sample_x = np.vstack((sample_x, lhs(1, 500) * 0.4 + ramp_loc[i + 1] - 0.2))
    sample_x = np.hstack((sample_x, lhs(1, 10000) * (24 - 24 / 288)))
    X_f_train = np.vstack((sample_x, lb + (ub - lb) * lhs(2, 10000)))
    N_f = region_N(X_f_train[:, 0:1])
    X_f_train = np.hstack((X_f_train, N_f))

    S = generate_S(flow, s_detect_id)

    model = PINN_JWZ(layers, lb, ub, ramp_loc=ramp_loc, ramp_type=ramp_type, device=device)
    model.to(device)
    scheduler = lr_scheduler.StepLR(model.optimizer, step_size=5000, gamma=0.97)

    # print parameter counts and FLOPs across ablations
    cfgs = [
        (True, True, True, "完整模型"),
        (False, True, True, "去掉注意力"),
        (True, False, True, "去掉参数网络"),
        (True, True, False, "去掉snet"),
    ]
    for inc_attn, inc_para, inc_snet, name in cfgs:
        total_params, trainable_params = count_parameters(model, include_attn=inc_attn, include_para=inc_para, include_snet=inc_snet)
        est_flops = estimate_flops_single_forward(model, include_attn=inc_attn, include_para=inc_para, include_snet=inc_snet)
        print(f"[{name}] 参数: 总 {total_params:,} | 训 {trainable_params:,} | FLOPs: {est_flops:,} (~{est_flops/1e6:.2f} MFLOPs)")

    x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True, dtype=torch.float32, device=device)
    t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True, dtype=torch.float32, device=device)
    n_u = torch.tensor(X_u_train[:, 2:3], requires_grad=True, dtype=torch.float32, device=device)

    x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True, dtype=torch.float32, device=device)
    t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True, dtype=torch.float32, device=device)
    n_f = torch.tensor(X_f_train[:, 2:3], requires_grad=True, dtype=torch.float32, device=device)

    flow_t = torch.tensor(u_train[:, 0:1], dtype=torch.float32, device=device)
    speed_t = torch.tensor(u_train[:, 1:2], dtype=torch.float32, device=device)

    t_z = np.linspace(0, 24 - 24 / 288, 288)
    X_z, T_z = np.meshgrid(x[s_detect_id], t_z)
    X_z_t = torch.tensor(X_z.reshape(-1, 1), requires_grad=True, dtype=torch.float32, device=device)
    T_z_t = torch.tensor(T_z.reshape(-1, 1), requires_grad=True, dtype=torch.float32, device=device)

    loc = torch.tensor(ramp_type, dtype=torch.float32, device=device)

    W1 = (60.0 / (speed_t + 30.0)).detach()
    X_F = x_f.repeat(1, 20) - torch.tensor(ramp_loc, dtype=torch.float32, device=device).reshape(1, -1)
    X_F_min = torch.min(torch.abs(X_F), dim=1)[0]
    W2 = torch.exp(X_F_min.reshape(-1, 1)).detach()

    best_loss = 1e9
    steps_since_improve = 0
    patience = 1000
    start_time = time.time()

    try:
        atomic_save_state_dict(model, save_path)
    except Exception:
        pass

    for step in range(total_steps):
        u_pred, f_pred, k_pred = model.net_u(x_u, t_u)
        phy1, phy2 = model.net_f(x_f, t_f, n_f)

        t_z_t = torch.tensor(t_z, dtype=torch.float32, device=device).reshape(-1, 1)
        _, f_S, _ = model.net_u(X_z_t, T_z_t)
        F = torch.sum(f_S.clone().detach().reshape(24, -1, len(s_detect_id)), dim=1)
        S_obs = F[:, 1:] - F[:, :-1]
        S_pred = torch.sum(model.generate_r(t_z_t).reshape(24, -1, 20), dim=1) * loc

        loss1 = torch.mean((f_pred - flow_t) ** 2 * W1) * alpha1
        loss2 = torch.mean(((u_pred - speed_t) ** 2) * W1) * alpha2
        loss3 = torch.mean((phy1 ** 2) * W2) * beta1
        loss4 = torch.mean((phy2 ** 2) * W2) * beta2
        loss5 = torch.mean((S_pred - S_obs) ** 2) * gamma
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            steps_since_improve = 0
            atomic_save_state_dict(model, save_path)
        else:
            steps_since_improve += 1

        if step % 100 == 0:
            pct = round(step / total_steps * 100)
            print(f"\r进度: {pct}% loss1:{loss1.item():.5f} loss2:{loss2.item():.5f} loss3:{loss3.item():.5f} loss4:{loss4.item():.5f} loss5:{loss5.item():.5f}", end="")
            sys.stdout.flush()

        if steps_since_improve >= patience:
            print(f"\nEarly stopping triggered at step {step} (no improvement for {patience} steps). Best loss: {best_loss:.6f}")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.2f}s")

    try:
        from .test import main_test
        main_test(data_dir=data_dir, weights_path=save_path)
    except Exception as e:
        print(f"Auto-test failed: {e}")


def env_or_default(key: str, default: str) -> str:
    val = os.environ.get(key)
    return val if val not in (None, "") else default


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=int(env_or_default("TOTAL_STEPS", "30000")))
    parser.add_argument("--save_path", type=str, default=env_or_default("SAVE_PATH", "best_model.pat"))
    parser.add_argument("--layers", type=str, default=env_or_default("LAYERS", "2,128,128,128,64,2"))
    parser.add_argument("--id_list", type=str, default=env_or_default("ID_LIST", "2,4,7,8,10,11,13,14,16,17,18,20,21,22,23,24,26,27,28,30,31"))
    parser.add_argument("--s_detect_id", type=str, default=env_or_default("S_DETECT_ID", "0,2,4,7,8,10,11,13,14,16,18,20,21,22,23,24,26,27,28,30,31"))
    parser.add_argument("--alpha1", type=float, default=float(env_or_default("ALPHA1", str(1/400))))
    parser.add_argument("--alpha2", type=float, default=float(env_or_default("ALPHA2", "60.0")))
    parser.add_argument("--beta1", type=float, default=float(env_or_default("BETA1", str(1/1000))))
    parser.add_argument("--beta2", type=float, default=float(env_or_default("BETA2", str(1/3000))))
    parser.add_argument("--gamma", type=float, default=float(env_or_default("GAMMA", str(1/2000))))
    args = parser.parse_args()

    data_root = os.path.dirname(os.path.abspath(__file__))
    resolved_data_dir = args.data_dir if args.data_dir not in (None, "") else os.path.join(data_root, "..")

    layers_parsed = parse_int_list_layers(args.layers, [2,128,128,128,64,2])
    id_list_parsed = parse_int_list_default(args.id_list, [2,4,7,8,10,11,13,14,16,17,18,20,21,22,23,24,26,27,28,30,31])
    s_detect_id_parsed = parse_int_list_default(args.s_detect_id, [0,2,4,7,8,10,11,13,14,16,18,20,21,22,23,24,26,27,28,30,31])

    main_train(
        data_dir=resolved_data_dir,
        total_steps=args.total_steps,
        save_path=args.save_path,
        layers=layers_parsed,
        id_list=id_list_parsed,
        s_detect_id=s_detect_id_parsed,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        beta1=args.beta1,
        beta2=args.beta2,
        gamma=args.gamma,
    )
