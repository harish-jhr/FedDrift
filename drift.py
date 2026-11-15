import torch

from diffusion import denoising_loss

# the five timestep bands we measure drift in
BUCKETS = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]


# gradient of the loss on one batch, flattened into one long vector
def flat_grad(model, x0, t, eps, sched):
    params = [p for p in model.parameters() if p.requires_grad]
    loss = denoising_loss(model, x0, t, eps, sched)
    grads = torch.autograd.grad(loss, params)
    return torch.cat([g.flatten() for g in grads])


def sq(x):
    return x.pow(2).sum(dtype=torch.float64).item()


# Measures how much the clients disagree in each timestep band.
#
# All clients start from the same model. In each band, every client computes a gradient on
# its own images, and all clients use the same timesteps and the same noise, so the only
# thing that differs between them is the images. Drift is how far the client gradients
# spread around their mean.
#
# A single minibatch gradient also carries the client's own sampling noise, which would look
# like disagreement. We remove it in two ways and keep both:
#   - zeta2_corrected: split each client's batch in two halves and subtract the variance
#     between the halves
#   - zeta2_cross: multiply deviations from different repeats, where the sampling noise
#     averages out (this is the number we report)
# The two agree to within a few percent.
def measure_drift(model, x_all, parts, sched, buckets=BUCKETS, batch=256, repeats=8, seed=0):
    device = x_all.device
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    K, R, half = len(parts), repeats, batch // 2

    results = []
    for lo, hi in buckets:
        raw = within = gbar2 = cos = 0.0
        sum_dev, sum_dev_sq = 0.0, 0.0
        sum_gbar, sum_gbar_sq = 0.0, 0.0

        for _ in range(R):
            # same timesteps and noise for every client in this repeat
            t = torch.randint(lo, hi, (half,), device=device, generator=gen)
            eps = torch.randn(half, *x_all.shape[1:], device=device, generator=gen)

            g = []
            for idx in parts:
                pick = idx[torch.randperm(len(idx), device=device, generator=gen)[:batch]]
                g1 = flat_grad(model, x_all[pick[:half]], t, eps, sched)
                g2 = flat_grad(model, x_all[pick[half:]], t, eps, sched)
                g.append((g1 + g2) / 2)
                within += sq(g1 - g2) / 4 / K

            g = torch.stack(g)
            gbar = g.mean(0)
            dev = g - gbar

            raw += sq(dev) / K
            gbar2 += sq(gbar)
            gn = g / g.norm(dim=1, keepdim=True)
            cos += ((gn @ gn.T).sum().item() - K) / (K * (K - 1))

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
