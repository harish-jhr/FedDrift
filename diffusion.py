import torch


# Noise schedule. t is an integer in [0, T), and bigger t means more noise.
#
# ddpm: the usual linear-beta DDPM. x_t = sqrt(abar) x0 + sqrt(1 - abar) eps, and the
#       network predicts the noise eps.
# fm:   flow matching (rectified flow). x_t = (1 - tau) x0 + tau eps with tau = (t + 1) / T,
#       and the network predicts the direction eps - x0.
#
# Both use the same integer t, so the timestep bands, the drift measurement and RegMean
# work the same way for either objective.
class Schedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu", kind="ddpm"):
        self.T, self.kind = T, kind
        if kind == "ddpm":
            self.betas = torch.linspace(beta_start, beta_end, T, device=device)
            self.alphas = 1.0 - self.betas
            self.alphabar = torch.cumprod(self.alphas, dim=0)
            self.sqrt_ab = self.alphabar.sqrt()
            self.sqrt_1mab = (1.0 - self.alphabar).sqrt()
        elif kind == "fm":
            self.tau = torch.arange(1, T + 1, device=device, dtype=torch.float32) / T
            # how much of the image and how much of the noise is left at each t
            self.sqrt_ab = 1.0 - self.tau
            self.sqrt_1mab = self.tau
            self.alphabar = self.sqrt_ab.pow(2)
        else:
            raise ValueError(kind)
        self.snr = self.sqrt_ab.pow(2) / self.sqrt_1mab.pow(2)


# add noise to clean images x0 at timesteps t
def q_sample(x0, t, eps, sched):
    return sched.sqrt_ab[t].view(-1, 1, 1, 1) * x0 + sched.sqrt_1mab[t].view(-1, 1, 1, 1) * eps


# what the network should predict: the noise for DDPM, the direction eps - x0 for flow matching
def target(x0, eps, sched):
    return eps if sched.kind == "ddpm" else eps - x0


# the training loss, the same for both objectives
def denoising_loss(model, x0, t, eps, sched):
    xt = q_sample(x0, t, eps, sched)
    return (model(xt, t) - target(x0, eps, sched)).pow(2).mean()


# Generates images by running the reverse process from pure noise.
# With clip=True we estimate the clean image at every step, clip it to [-1, 1] and step from
# there (the standard sampler from Ho et al.). Without it, 64x64 samples drift in colour after
# about t = 700 and come out as one flat colour. By default we only clip above 32x32, so the
# CIFAR-10 and MNIST numbers stay comparable with our earlier runs.
@torch.no_grad()
def ddpm_sample(model, shape, sched, device, gen=None, clip=None):
    if sched.kind == "fm":
        return fm_sample(model, shape, sched, device, gen)
    if clip is None:
        clip = shape[-1] > 32
    x = torch.randn(shape, device=device, generator=gen)
    for i in reversed(range(sched.T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t).float()
        if clip:
            ab, ab_prev = sched.alphabar[i], (sched.alphabar[i - 1] if i > 0 else torch.ones((), device=device))
            x0 = ((x - sched.sqrt_1mab[i] * eps) / sched.sqrt_ab[i]).clamp(-1, 1)
            mean = (ab_prev.sqrt() * sched.betas[i] * x0 + sched.alphas[i].sqrt() * (1 - ab_prev) * x) / (1 - ab)
            var = sched.betas[i] * (1 - ab_prev) / (1 - ab)
        else:
            mean = (x - sched.betas[i] / sched.sqrt_1mab[i] * eps) / sched.alphas[i].sqrt()
            var = sched.betas[i]
        noise = torch.randn(shape, device=device, generator=gen) if i > 0 else torch.zeros_like(x)
        x = mean + var.sqrt() * noise
    return x


# flow matching sampler: simple Euler steps from tau = 1 (noise) down to 0 (image),
# one step per t, so it uses the same 1000 network calls as DDPM
@torch.no_grad()
def fm_sample(model, shape, sched, device, gen=None):
    x = torch.randn(shape, device=device, generator=gen)
    for i in reversed(range(sched.T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        v = model(x, t).float()
        tau_next = sched.tau[i - 1] if i > 0 else 0.0
        x = x + (tau_next - sched.tau[i]) * v
    return x
