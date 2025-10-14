import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN_JWZ(nn.Module):
    def __init__(self, layers, lb, ub, ramp_loc, ramp_type, device=None):
        super().__init__()
        self.device = device or get_device()
        self.lb = torch.tensor(lb, dtype=torch.float32, device=self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=self.device)
        self.ramp_loc = torch.tensor(ramp_loc, dtype=torch.float32, device=self.device)
        self.ramp_type = torch.tensor(ramp_type, dtype=torch.float32, device=self.device)

        self.fc1 = nn.Linear(8, layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], layers[3])
        self.fc4 = nn.Linear(layers[3], layers[4])
        self.fc5 = nn.Linear(8, layers[1])
        self.fc6 = nn.Linear(layers[1], layers[2])
        self.fc7 = nn.Linear(layers[2], layers[3])
        self.fc8 = nn.Linear(layers[3], layers[4])
        self.fc9 = nn.Linear(layers[4], 1)
        self.fc10 = nn.Linear(layers[4], 1)

        for m in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8, self.fc9, self.fc10]:
            nn.init.xavier_normal_(m.weight)

        def att(in_dim: int):
            return nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        self.att_fc1 = att(layers[1])
        self.att_fc2 = att(layers[2])
        self.att_fc3 = att(layers[3])
        self.att_fc4 = att(layers[4])
        self.att_fc5 = att(layers[1])
        self.att_fc6 = att(layers[2])
        self.att_fc7 = att(layers[3])
        self.att_fc8 = att(layers[4])

        self.snet = nn.Sequential(
            nn.Linear(1, layers[1]), nn.Tanh(),
            nn.Linear(layers[1], layers[2]), nn.Tanh(),
            nn.Linear(layers[2], layers[3]), nn.Tanh(),
            nn.Linear(layers[3], layers[4]), nn.Tanh(),
            nn.Linear(layers[4], 20), nn.Sigmoid(),
        )
        self.para = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 7), nn.Sigmoid(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.97)

    def _feature(self, x, t):
        X = torch.cat([x, t], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        H = torch.cat([H, torch.cos(H), torch.sin(H), torch.exp(H)], dim=1)
        return H.float()

    def net_u(self, x, t):
        H = self._feature(x.float(), t.float())
        u1 = torch.tanh(self.fc1(H))
        a1 = self.att_fc1(u1)
        u1 = u1 * a1
        u2 = torch.tanh(self.fc2(u1))
        a2 = self.att_fc2(u2)
        u2 = u2 * a2
        u3 = torch.tanh(self.fc3(u2))
        a3 = self.att_fc3(u3)
        u3 = u3 * a3
        u4 = torch.tanh(self.fc4(u3))
        a4 = self.att_fc4(u4)
        u4 = u4 * a4

        v1 = torch.tanh(self.fc5(H))
        b1 = self.att_fc5(v1)
        v1 = v1 * b1
        v2 = torch.tanh(self.fc6(v1))
        b2 = self.att_fc6(v2)
        v2 = v2 * b2
        v3 = torch.tanh(self.fc7(v2))
        b3 = self.att_fc7(v3)
        v3 = v3 * b3
        v4 = torch.tanh(self.fc8(v3))
        b4 = self.att_fc8(v4)
        v4 = v4 * b4

        u = torch.sigmoid(self.fc9(u4)) * 100 + 10
        f = torch.sigmoid(self.fc10(v4)) * 6000
        k = f / u
        return u, f, k

    def generate_r(self, t):
        H = (2 * t.float() / 24.0 - 1.0)
        y = self.snet(H) * 2000
        return y

    def net_s(self, x, t):
        y = self.generate_r(t)
        s = torch.zeros_like(x, device=self.device)
        for i in range(20):
            loc = self.ramp_loc[i]
            s = s + (1.0 / (torch.sqrt(torch.tensor(2.0 * torch.pi, device=self.device)) * 0.1) *
                     torch.exp(-((x - loc) ** 2) / (2 * 0.1 ** 2)) * y[:, i].reshape(-1, 1) * self.ramp_type[i])
        return s

    def get_para(self, x, t):
        X = torch.cat([x.float(), t.float()], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        para = self.para(H.float())
        v_f = para[:, 0] * 60 + 60
        k_c = para[:, 1] * 50 + 15
        alpha = para[:, 2] * 0.4 + 2.1
        sigma = para[:, 3] * 2 + 0.001
        tau = para[:, 4] / 30 + 1 / 3600
        c = para[:, 5] * 80 + 1
        k_min = para[:, 6] * 20 + 5
        return v_f.reshape(-1, 1), k_c.reshape(-1, 1), alpha.reshape(-1, 1), sigma.reshape(-1, 1), tau.reshape(-1, 1), c.reshape(-1, 1), k_min.reshape(-1, 1)

    def net_f(self, x, t, n):
        u, f, k = self.net_u(x, t)
        s = self.net_s(x, t)
        v_f, k_c, alpha, sigma, tau, c, k_min = self.get_para(x, t)
        k_t = torch.autograd.grad(k.sum(), t, create_graph=True)[0]
        f_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        k_x = torch.autograd.grad(k.sum(), x, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        p = k / n
        U = v_f * torch.exp(-((p / k_c) ** alpha) / alpha)
        phy = (k_t + f_x - s)
        phy1 = u_t + u * u_x - c * u_x - (U - u) / tau + sigma * torch.abs(s) * u / (k + k_min * n)
        return phy, phy1

    def forward(self, x, t):
        return self.net_u(x, t)

    def predict(self, X_star):
        x = torch.tensor(X_star[:, 0:1], requires_grad=True, dtype=torch.float32, device=self.device)
        t = torch.tensor(X_star[:, 1:2], requires_grad=True, dtype=torch.float32, device=self.device)
        u, f, _ = self.net_u(x, t)
        return u.detach().cpu().numpy(), f.detach().cpu().numpy()


def estimate_flops_single_forward(model: PINN_JWZ, include_attn: bool = True, include_para: bool = True, include_snet: bool = True) -> int:
    flops = 0
    flops += 2 * 4
    flops += 8 * 3

    def lin_flops(in_f, out_f):
        return in_f * out_f * 2

    flops += lin_flops(8, model.fc1.out_features)
    flops += lin_flops(model.fc1.out_features, model.fc2.out_features)
    flops += lin_flops(model.fc2.out_features, model.fc3.out_features)
    flops += lin_flops(model.fc3.out_features, model.fc4.out_features)

    flops += lin_flops(8, model.fc5.out_features)
    flops += lin_flops(model.fc5.out_features, model.fc6.out_features)
    flops += lin_flops(model.fc6.out_features, model.fc7.out_features)
    flops += lin_flops(model.fc7.out_features, model.fc8.out_features)

    if include_attn:
        def att_flops(in_f):
            return lin_flops(in_f, 16) + lin_flops(16, 1)
        flops += att_flops(model.fc1.out_features)
        flops += att_flops(model.fc2.out_features)
        flops += att_flops(model.fc3.out_features)
        flops += att_flops(model.fc4.out_features)
        flops += att_flops(model.fc5.out_features)
        flops += att_flops(model.fc6.out_features)
        flops += att_flops(model.fc7.out_features)
        flops += att_flops(model.fc8.out_features)

    flops += lin_flops(model.fc4.out_features, 1)
    flops += lin_flops(model.fc8.out_features, 1)

    l1, l2, l3, l4 = model.fc1.out_features, model.fc2.out_features, model.fc3.out_features, model.fc4.out_features
    if include_snet:
        flops += lin_flops(1, l1)
        flops += lin_flops(l1, l2)
        flops += lin_flops(l2, l3)
        flops += lin_flops(l3, l4)
        flops += lin_flops(l4, 20)

    if include_para:
        flops += lin_flops(2, 64)
        flops += lin_flops(64, 128)
        flops += lin_flops(128, 7)

    return int(flops)


def count_parameters(model: PINN_JWZ, include_attn: bool = True, include_para: bool = True, include_snet: bool = True) -> tuple[int, int]:
    total = 0
    trainable = 0

    def add_params(mod: nn.Module):
        nonlocal total, trainable
        for p in mod.parameters(recurse=True):
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n

    # always include main layers and outputs
    for m in [model.fc1, model.fc2, model.fc3, model.fc4, model.fc5, model.fc6, model.fc7, model.fc8, model.fc9, model.fc10]:
        add_params(m)

    if include_attn:
        for m in [model.att_fc1, model.att_fc2, model.att_fc3, model.att_fc4, model.att_fc5, model.att_fc6, model.att_fc7, model.att_fc8]:
            add_params(m)
    if include_snet:
        add_params(model.snet)
    if include_para:
        add_params(model.para)

    return total, trainable
