import os

import numpy as np
import torch
from torchvision import datasets


def load_cifar10_uint8(root="./data", split="train"):
    #raw CIFAR-10 pixels as N x 3 x 32 x 32 uint8, no normalisation. this is the FID reference set
    ds = datasets.CIFAR10(root, train=(split == "train"), download=True)
    return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()


def to_uint8(x):
    #[-1,1] float -> uint8, same rounding as saving the samples as PNGs. both the generated and the
    #real images go through Inception as uint8, so quantisation is not a difference between them
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def gaussian_stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def psd_sqrt(a):
    w, v = np.linalg.eigh((a + a.T) / 2)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def frechet_distance(mu, sigma, mu_ref, sigma_ref, sqrt_ref):
    # ||mu - mu_r||^2 + tr(S) + tr(S_r) - 2 tr((S S_r)^(1/2)).
    # tr((S S_r)^(1/2)) = sum sqrt(eig(R S R)) with R = S_r^(1/2), and R S R is symmetric PSD, so this
    # needs only eigvalsh. no scipy.linalg.sqrtm and no complex-valued intermediates to blow up
    # on the degenerate covariances you get from garbage samples at round 0
    m = sqrt_ref @ sigma @ sqrt_ref
    lam = np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0.0, None)
    return float(((mu - mu_ref) ** 2).sum() + np.trace(sigma) + np.trace(sigma_ref) - 2.0 * np.sqrt(lam).sum())


class FIDScorer:
    """FID against a fixed real-image reference set, using the pytorch-fid port of the TF Inception
    weights (pool3, 2048-d). Reference statistics are computed once and cached."""

    def __init__(self, device, data_root="./data", ref_split="train", cache_dir=".", batch_size=32):
        from pytorch_fid.inception import InceptionV3
        self.device, self.batch_size = device, batch_size
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True,
                               normalize_input=True).to(device).eval()

        cache = os.path.join(cache_dir, f"fid_ref_cifar10_{ref_split}.npz")
        n_expected = 50000 if ref_split == "train" else 10000
        if os.path.exists(cache):
            z = np.load(cache)
            assert int(z["n"]) == n_expected, f"{cache} has {int(z['n'])} images, expected {n_expected}"
            self.mu, self.sigma = z["mu"], z["sigma"]
        else:
            imgs = load_cifar10_uint8(data_root, ref_split)
            assert len(imgs) == n_expected
            print(f"computing FID reference stats on CIFAR-10 {ref_split} ({len(imgs)} images)...")
            self.mu, self.sigma = gaussian_stats(self.features(imgs))
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache, mu=self.mu, sigma=self.sigma, n=len(imgs))
        self.n_ref = n_expected
        self.sqrt_ref = psd_sqrt(self.sigma)

    @torch.inference_mode()
    def features(self, imgs_u8):
        feats = []
        for i in range(0, len(imgs_u8), self.batch_size):
            b = imgs_u8[i:i + self.batch_size].to(self.device).float() / 255.0
            feats.append(self.net(b)[0].flatten(1).float().cpu())
            del b
            if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(feats).numpy()

    def score(self, imgs_u8):
        mu, sigma = gaussian_stats(self.features(imgs_u8))
        return frechet_distance(mu, sigma, self.mu, self.sigma, self.sqrt_ref)
