import json
import os

import numpy as np
import torch
from torchvision import datasets

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

DATASETS = ["cifar10", "mnist", "celeba64", "tinyimagenet"]
IMG_SIZE = {"cifar10": 32, "mnist": 28, "celeba64": 64, "tinyimagenet": 64}
CHANNELS = {"cifar10": 3, "mnist": 1, "celeba64": 3, "tinyimagenet": 3}
# CelebA has no classes, so clients are skewed over the 2^3 groups of these attributes
CELEBA_ATTRS = ("Male", "Smiling", "Black_Hair")
# CelebA and Tiny-ImageNet are subsampled to CIFAR-10's size so client sizes stay comparable
N_TRAIN = 50000
N_TEST = 10000

# horizontal flips are turned off for MNIST (train.py sets this)
FLIP = True


def to_unit(x):
    return x.float() / 127.5 - 1.0


def load_cifar10(root="./data", train=True):
    ds = datasets.CIFAR10(root, train=train, download=True)
    return to_unit(torch.from_numpy(ds.data).permute(0, 3, 1, 2)), np.array(ds.targets)


def load_cifar100(root="./data", train=True):
    ds = datasets.CIFAR100(root, train=train, download=True)
    return to_unit(torch.from_numpy(ds.data).permute(0, 3, 1, 2)), np.array(ds.targets)


def load_mnist(root="./data", train=True):
    ds = datasets.MNIST(root, train=train, download=True)
    return to_unit(ds.data.unsqueeze(1)), ds.targets.numpy()


# Images in [-1, 1] as a float tensor [N, C, H, W], plus integer labels.
def load_dataset(name, root="./data", train=True):
    if name == "cifar10":
        return load_cifar10(root, train)
    if name == "mnist":
        return load_mnist(root, train)
    path = os.path.join(root, f"{name}_{'train' if train else 'test'}.npz")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        x, labels = (fetch_celeba if name == "celeba64" else fetch_tinyimagenet)(train)
        np.savez(path + ".tmp.npz", x=x, labels=labels)
        os.replace(path + ".tmp.npz", path)
    d = np.load(path)
    return to_unit(torch.from_numpy(d["x"]).permute(0, 3, 1, 2)), d["labels"]


def fetch_celeba(train):
    from datasets import load_dataset as hf_load
    from PIL import Image
    n = N_TRAIN if train else N_TEST
    split = "train" if train else "test"
    # rows are sorted by identity, so we keep a seeded random subset of the whole split
    total = {"train": 162770, "test": 19962}[split]
    keep = np.zeros(total, dtype=bool)
    keep[np.random.default_rng(0).choice(total, min(n, total), replace=False)] = True
    xs, labels = [], []
    for i, ex in enumerate(hf_load("flwrlabs/celeba", split=split, streaming=True)):
        if i >= total or len(xs) == n:
            break
        if not keep[i]:
            continue
        im = ex["image"].convert("RGB")
        w, h = im.size
        l, t = (w - 140) // 2, (h - 140) // 2
        xs.append(np.asarray(im.crop((l, t, l + 140, t + 140)).resize((64, 64), Image.LANCZOS)))
        labels.append(sum(int(ex[a]) << j for j, a in enumerate(reversed(CELEBA_ATTRS))))
    return np.stack(xs), np.array(labels)


def fetch_tinyimagenet(train):
    from datasets import load_dataset as hf_load
    ds = hf_load("zh-plus/tiny-imagenet", split="train" if train else "valid")
    labels = np.array(ds["label"])
    keep = np.arange(len(labels))
    if train and len(keep) > N_TRAIN:
        rng = np.random.default_rng(0)
        per = N_TRAIN // len(np.unique(labels))
        keep = np.sort(np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:per] for c in np.unique(labels)]))
    return np.stack([np.asarray(ds[int(i)]["image"].convert("RGB")) for i in keep]), labels[keep]


def random_flip(x):
    if not FLIP:
        return x
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)


# Splits the dataset between clients.
#   alpha = None     IID, every client gets a random equal share
#   alpha = "class"  one class per client (or #classes / #clients classes each)
#   alpha = number   Dirichlet split over classes, smaller alpha means more skew. We redraw
#                    until every client has at least min_size images.
def dirichlet_partition(labels, n_clients, alpha, seed, min_size=256, max_tries=10000):
    rng = np.random.default_rng(seed)

    if alpha is None:
        perm = rng.permutation(len(labels))
        return [np.sort(p) for p in np.array_split(perm, n_clients)]

    if alpha == "class":
        classes = np.unique(labels)
        if len(classes) % n_clients:
            raise ValueError(f"the class split needs #classes ({len(classes)}) divisible by #clients ({n_clients})")
        return [np.flatnonzero(np.isin(labels, classes[k::n_clients])) for k in range(n_clients)]

    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == c))
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
        parts = [np.sort(np.concatenate(p)) for p in parts]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError(f"could not get every client above {min_size} samples")


def count_matrix(labels, parts):
    return np.stack([np.bincount(labels[p], minlength=labels.max() + 1) for p in parts])


def print_counts(counts, title=""):
    print(title)
    if counts.shape[1] != len(CLASSES):
        for k, row in enumerate(counts):
            print(f"client {k:>2}: {row.sum():>6} images, {(row > 0).sum():>3} classes, top-class share {row.max() / row.sum():.2f}")
        print()
        return
    print("client " + "".join(f"{c[:5]:>6}" for c in CLASSES) + "   total")
    for k, row in enumerate(counts):
        print(f"{k:>6} " + "".join(f"{v:>6}" for v in row) + f"{row.sum():>8}")
    print()


def alpha_tag(alpha):
    if alpha is None:
        return "iid"
    return alpha if alpha == "class" else f"{alpha:g}"


def parse_alpha(s):
    if s == "iid":
        return None
    return s if s == "class" else float(s)


def partition_path(out_dir, n_clients, alpha, seed, dataset="cifar10"):
    tag = alpha_tag(alpha)
    tag = tag if tag in ("iid", "class") else f"alpha{tag}"
    return f"{out_dir}/{dataset}_K{n_clients}_{tag}_seed{seed}.json"


def save_partition(path, parts, alpha, seed):
    with open(path, "w") as f:
        json.dump({"alpha": alpha, "seed": seed, "parts": [p.tolist() for p in parts]}, f)


def load_partition(path):
    with open(path) as f:
        return [np.array(p) for p in json.load(f)["parts"]]
