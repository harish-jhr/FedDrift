# Helpers shared by the one-round tests: flatten and set weights, run local steps,
# and one FedAvg round from a fixed starting point.
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as p2v


def flat(model):
    return p2v(list(model.parameters())).detach().clone()


@torch.no_grad()
def setflat(model, w):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(w[i:i + n].view_as(p))
        i += n


def cos(a, b):
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def hvp(grad_fn, w, v, batch, rel=1e-3):
    r = rel * max(w.norm().item(), 1.0) / (v.norm().item() + 1e-30)
    gp = grad_fn(w + r * v, batch)
    gm = grad_fn(w - r * v, batch)
    return (gp - gm) / (2 * r)


def run_steps(model, w0, steps, step_loss, lr, opt_kind, use_amp, clip=1.0):
    setflat(model, w0)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr) if opt_kind == "adam" else torch.optim.SGD(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=use_amp):
            loss = step_loss(model, k)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
    v = None
    if opt_kind == "adam":
        st = opt.state_dict()["state"]
        b2 = opt.param_groups[0]["betas"][1]
        v = torch.cat([st[i]["exp_avg_sq"].flatten() for i in range(len(st))]) / (1 - b2 ** steps)
    return flat(model) - w0, v


def fedavg_round(model, w0, client_steps, pi, lr, opt_kind, use_amp, clip=1.0):
    deltas, vs = [], []
    for step_loss, steps in client_steps:
        d, v = run_steps(model, w0, steps, step_loss, lr, opt_kind, use_amp, clip)
        deltas.append(d)
        vs.append(v)
    return sum(p * d for p, d in zip(pi, deltas)), deltas, vs
