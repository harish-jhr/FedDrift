import torch


class Schedule:
    #linear beta schedule from DDPM, T=1000
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphabar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = self.alphabar.sqrt()
        self.sqrt_1mab = (1.0 - self.alphabar).sqrt()
        self.snr = self.alphabar / (1.0 - self.alphabar)


def q_sample(x0, t, eps, sched):
    return sched.sqrt_ab[t].view(-1, 1, 1, 1) * x0 + sched.sqrt_1mab[t].view(-1, 1, 1, 1) * eps


def ddpm_loss(model, x0, t, eps, sched):
    # t and eps are passed in, not sampled here.
    # the drift measurement has to feed the exact same t and eps to every client
    xt = q_sample(x0, t, eps, sched)
    return (model(xt, t) - eps).pow(2).mean()


@torch.no_grad()
def ddpm_sample(model, shape, sched, device):
    #plain sampling, all T steps
    x = torch.randn(shape, device=device)
    for i in reversed(range(sched.T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t)
        mean = (x - sched.betas[i] / sched.sqrt_1mab[i] * eps) / sched.alphas[i].sqrt()
        noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = mean + sched.betas[i].sqrt() * noise
    return x
