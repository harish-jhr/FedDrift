import math

import torch
import torch.nn as nn

from diffusion import q_sample

# RegMean merging with a separate strength for each timestep band.
#
# FedAvg just averages the client weights. RegMean instead picks, for each layer, the weights
# whose outputs best match every client's outputs on that client's own inputs. The answer only
# needs the Gram matrix of each layer's inputs (G = X^T X), not the inputs themselves, so no
# images ever leave a client.
#
# For one layer, with client updates d_k, client data shares pi_k and Gram matrices G_kb
# (client k, band b), the merged update is
#
#   u = ( dbar S + sum_b rho_b ( sum_k pi_k d_k G_kb - dbar Gbar_b ) ) (S + ridge)^-1
#
# where dbar is the FedAvg update, Gbar_b = sum_k pi_k G_kb and S = sum_b Gbar_b.
# The first term alone gives back FedAvg, and each band adds its own correction scaled by
# rho_b. rho_b = 1 everywhere is plain RegMean, rho_b = 0 everywhere is FedAvg.


# every conv and linear layer, these are the ones RegMean merges
def layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


# Gram matrix of every layer's inputs, one per timestep band.
# We push n noisy copies of the client's own images through the model for each band and
# record the inputs with forward hooks. For convolutions we use the channel-level Gram
# (inputs summed over image positions), so each Gram is only channels x channels.
# Returns {layer name: tensor of shape [n_bands, d_in, d_in]}.
@torch.no_grad()
def grams(model, x, sched, bands, n, batch, gen):
    model.eval()
    mods = layers(model)
    out = {name: [] for name, _ in mods}
    cur, cnt = {}, {}

    def hook(name):
        def f(m, inp):
            a = inp[0].float()
            if a.dim() == 4:
                a = a.movedim(1, -1)
            a = a.reshape(-1, a.shape[-1])
            cur[name] = cur.get(name, 0) + a.T @ a
            cnt[name] = cnt.get(name, 0) + a.shape[0]
        return f

    handles = [m.register_forward_pre_hook(hook(name)) for name, m in mods]
    for lo, hi in bands:
        cur.clear()
        cnt.clear()
        for i in range(0, n, batch):
            b = min(batch, n - i)
            idx = torch.randint(0, len(x), (b,), device=x.device, generator=gen)
            t = torch.randint(lo, hi, (b,), device=x.device, generator=gen)
            eps = torch.randn((b, *x.shape[1:]), device=x.device, generator=gen)
            model(q_sample(x[idx], t, eps, sched), t)
        for name in out:
            out[name].append(cur[name] / cnt[name])
    for h in handles:
        h.remove()
    return {name: torch.stack(v) for name, v in out.items()}


# plain RegMean for one layer (all bands together), used by the one-round tests in experiments/
def regmean_layer(ds, gs, pi, a, rel):
    gs = [a * g + (1 - a) * torch.diag(torch.diagonal(g)) for g in gs]
    S = sum(p * g for p, g in zip(pi, gs))
    S = S + rel * S.diagonal().mean() * torch.eye(len(S), device=S.device)
    N = sum(p * (d.movedim(1, -1) @ g) for p, d, g in zip(pi, ds, gs))
    X = torch.linalg.solve(S, N.reshape(-1, S.shape[0]).T).T
    return X.reshape(N.shape).movedim(-1, 1)


# Adaptive band strengths.
# gain[b] is how much local training lowered the loss in band b on held-out images of the
# client's own class, averaged over clients. If local training does not carry over to new
# images in a band, that band's correction should be weaker. So rho_b grows like
# exp(gain_b / tau), with a floor of 0.2. Then we pick one common scale (by bisection) so that,
# after clipping to [0.2, 2], the strengths have mean `strength`.
def band_strengths(gain, tau=0.05, clip=(0.2, 2.0), strength=1.0):
    nb = len(gain)
    raw = [max(clip[0], math.exp(min(g / tau, 50.0))) for g in gain]
    clipped = lambda c: [min(clip[1], max(clip[0], c * v)) for v in raw]
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(clipped(math.exp(mid))) < nb else (lo, mid)
    rho = clipped(math.exp((lo + hi) / 2))
    m = sum(rho) / nb
    return [strength * v / m for v in rho]


# The band-weighted RegMean merge for one round.
#
# prev    global weights before the round
# states  client weights after local training
# G       one grams() output per client
# pi      client data shares
# rho     strength per band
# a       shrinks the off-diagonal Gram entries (G <- a G + (1 - a) diag(G)), we use 0.9
# rel     small ridge, as a fraction of the mean diagonal, so the solve is stable
# beta    memory term. With beta < 1, part of RegMean's extra correction repeats every round
#         and keeps piling up in the weights, so we keep a fading memory C of it:
#             model = prev + u - (1 - beta) C,   C <- beta C + (u - dbar)
#         `memory` is the dict that holds C between rounds. beta = 1 switches it off.
#
# Timestep-embedding layers are skipped and left to FedAvg. They only ever see the timestep,
# so their Gram matrices are nearly singular, and inverting them blew the update up 175x.
# Returns the new weights for the merged layers, plus update sizes for logging.
def merge(prev, states, G, pi, rho, a=0.9, rel=1e-4, beta=1.0, memory=None):
    nb = len(rho)
    out = {}
    stats = {"u": 0.0, "fedavg_u": 0.0, "applied_u": 0.0, "C_norm": 0.0, "max_layer_ratio": 0.0}
    for name in G[0]:
        if "time" in name:
            continue
        key = name + ".weight"
        ds = [(st[key].float() - prev[key].float()).movedim(1, -1) for st in states]
        gs = [[a * g[name][b] + (1 - a) * torch.diag(torch.diagonal(g[name][b])) for b in range(nb)] for g in G]
        bar = [sum(p * gk[b] for p, gk in zip(pi, gs)) for b in range(nb)]
        S = sum(bar)
        Sr = S + rel * S.diagonal().mean() * torch.eye(len(S), device=S.device)
        dbar = sum(p * d for p, d in zip(pi, ds))

        # FedAvg part, then one correction per band scaled by its strength
        N = dbar @ S
        for b in range(nb):
            N = N + rho[b] * (sum(p * (d @ gk[b]) for p, d, gk in zip(pi, ds, gs)) - dbar @ bar[b])
        u = torch.linalg.solve(Sr, N.reshape(-1, Sr.shape[0]).T).T.reshape(N.shape).movedim(-1, 1)
        fa = dbar.movedim(-1, 1)

        applied = u
        if beta < 1.0:
            C = memory.get(key)
            if C is not None:
                applied = u - (1 - beta) * C
            memory[key] = (u - fa) if C is None else beta * C + (u - fa)
            stats["C_norm"] += memory[key].pow(2).sum().item()
        stats["u"] += u.pow(2).sum().item()
        stats["fedavg_u"] += fa.pow(2).sum().item()
        stats["applied_u"] += applied.pow(2).sum().item()
        stats["max_layer_ratio"] = max(stats["max_layer_ratio"], (u.norm() / (fa.norm() + 1e-12)).item())
        out[key] = (prev[key].float() + applied).to(prev[key].dtype)
    for k in ("u", "fedavg_u", "applied_u", "C_norm"):
        stats[k] **= 0.5
    return out, stats
