import torch

from diffusion import ddpm_loss

#5 uniform buckets in t
BUCKETS = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]


def flat_grad(model, x0, t, eps, sched):
    params = [p for p in model.parameters() if p.requires_grad]
    loss = ddpm_loss(model, x0, t, eps, sched)
    grads = torch.autograd.grad(loss, params)
    return torch.cat([g.flatten() for g in grads])


def sq(x):
    # squared norm in float64, we subtract big nearly-equal numbers later
    return x.pow(2).sum(dtype=torch.float64).item()


def measure_drift(model, x_all, parts, sched, buckets=BUCKETS, batch=256, repeats=8, seed=0):
    """Client gradient dissimilarity zeta^2 = E_i ||g_i - g_bar||^2, separately per timestep bucket.

    model  : the GLOBAL model. every client's gradient is taken at these same weights,
             zeta^2 is only defined at a common point
    x_all  : full dataset on device, un-augmented
    parts  : list of index tensors, one per client
    """
    device = x_all.device
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    K, R, half = len(parts), repeats, batch // 2

    results = []
    for lo, hi in buckets:
        raw = within = gbar2 = cos = 0.0
        # running sums over repeats, for the cross-repeat estimator below
        sum_dev, sum_dev_sq = 0.0, 0.0
        sum_gbar, sum_gbar_sq = 0.0, 0.0

        for _ in range(R):
            # common random numbers: ONE t and ONE eps, reused by every client and both halves.
            # the only thing that differs between two gradients is then the data
            t = torch.randint(lo, hi, (half,), device=device, generator=gen)
            eps = torch.randn(half, *x_all.shape[1:], device=device, generator=gen)

            g = []
            for idx in parts:
                pick = idx[torch.randperm(len(idx), device=device, generator=gen)[:batch]]
                g1 = flat_grad(model, x_all[pick[:half]], t, eps, sched)
                g2 = flat_grad(model, x_all[pick[half:]], t, eps, sched)
                g.append((g1 + g2) / 2)
                # the two halves are the same client, so they differ only by minibatch noise.
                # ||g1-g2||^2 / 4 estimates the noise variance of their average
                within += sq(g1 - g2) / 4 / K

            g = torch.stack(g)                    # K x n_params
            gbar = g.mean(0)                      # every client counts equally
            dev = g - gbar

            raw += sq(dev) / K
            gbar2 += sq(gbar)
            gn = g / g.norm(dim=1, keepdim=True)
            cos += ((gn @ gn.T).sum().item() - K) / (K * (K - 1))   # mean over pairs i != j

            sum_dev = sum_dev + dev
            sum_dev_sq += sq(dev)
            sum_gbar = sum_gbar + gbar
            sum_gbar_sq += sq(gbar)

        raw, within, gbar2, cos = raw / R, within / R, gbar2 / R, cos / R
        corrected = max(raw - (K - 1) / K * within, 0.0)
        cross = (sq(sum_dev) - sum_dev_sq) / (K * R * (R - 1))
        gbar2_cross = (sq(sum_gbar) - sum_gbar_sq) / (R * (R - 1))

        mid = (lo + hi) // 2
        results.append({
            "bucket": [lo, hi],
            "snr_mid": sched.snr[mid].item(),
            "zeta2_raw": raw,
            "zeta2_corrected": corrected,
            "zeta2_cross": cross,
            "zeta2_normalized": corrected / gbar2,
            "zeta2_cross_normalized": cross / gbar2_cross,
            "within_client_var": within,
            "mean_grad_norm2": gbar2,
            "mean_pairwise_cos": cos,
        })
    return results
