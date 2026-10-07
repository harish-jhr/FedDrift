import torch
import torch.nn as nn

from data import random_flip
from diffusion import ddpm_loss


def local_train(model, x_all, idx, sched, epochs, batch_size, lr, grad_clip=1.0):
    #client's local training, x_all is the full dataset on device,idx this client's indices
    device = x_all.device
    use_amp = device.type == "cuda"
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total, steps = 0.0, 0
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx), device=device)]
        for i in range(0, len(perm), batch_size):
            x0 = random_flip(x_all[perm[i:i + batch_size]])
            t = torch.randint(0, sched.T, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = ddpm_loss(model, x0, t, eps, sched)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            total += loss.item()
            steps += 1
    return total / max(steps, 1), steps


def fedavg(states, sizes):
    #average client weights, weighted by how much data each client has
    w = torch.tensor(sizes, dtype=torch.float64)
    w = w / w.sum()
    avg = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            avg[k] = sum(wi.item() * s[k] for wi, s in zip(w, states))
        else:
            avg[k] = v.clone()
    return avg
