import json
import numpy as np
import torch
from torchvision import datasets

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def load_cifar10(root="./data"):
    #loads CIFAR-10 as a single tensor and normalizes to [-1,1] rnge
    ds = datasets.CIFAR10(root, train=True, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    labels = np.array(ds.targets)
    return x, labels


def random_flip(x):
    #randmomly flip some images, std aug for local training
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)


def dirichlet_partition(labels, n_clients, alpha, seed, min_size=256, max_tries=10000):
    rng = np.random.default_rng(seed)

    # alpha=None is the IID control: shuffle and chunk
    if alpha is None:
        perm = rng.permutation(len(labels))
        return [np.sort(p) for p in np.array_split(perm, n_clients)]

    for _ in range(max_tries):
        #sample a dirichlet dist for each class, then assign each sample to a client according to the class distribution
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == c))
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
        parts = [np.sort(np.concatenate(p)) for p in parts]
        #reddo if smallest split is smaller than permissible min_size
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError(f"could not get every client above {min_size} samples")


def count_matrix(labels, parts):
    return np.stack([np.bincount(labels[p], minlength=len(CLASSES)) for p in parts])


def print_counts(counts, title=""):
    print(title)
    print("client " + "".join(f"{c[:5]:>6}" for c in CLASSES) + "   total")
    for k, row in enumerate(counts):
        print(f"{k:>6} " + "".join(f"{v:>6}" for v in row) + f"{row.sum():>8}")
    print()


def partition_path(out_dir, n_clients, alpha, seed):
    tag = "iid" if alpha is None else f"alpha{alpha:g}"
    return f"{out_dir}/cifar10_K{n_clients}_{tag}_seed{seed}.json"


def save_partition(path, parts, alpha, seed):
    with open(path, "w") as f:
        json.dump({"alpha": alpha, "seed": seed, "parts": [p.tolist() for p in parts]}, f)


def load_partition(path):
    with open(path) as f:
        return [np.array(p) for p in json.load(f)["parts"]]
