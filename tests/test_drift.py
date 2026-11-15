import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data import load_cifar10
from diffusion import Schedule
from drift import measure_drift
from model import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    x_all, labels = load_cifar10(args.data_root)
    x_all = x_all.to(device)
    sched = Schedule(device=device)
    model = UNet("small").to(device)

    cats = np.flatnonzero(labels == 3)
    dogs = np.flatnonzero(labels == 5)
    mixed = np.random.default_rng(0).permutation(len(labels))[:10000]
    disjoint = [torch.from_numpy(p).to(device) for p in (cats, dogs)]
    iid = [torch.from_numpy(p).to(device) for p in (mixed[:5000], mixed[5000:])]

    buckets = [(0, 200), (400, 600), (800, 1000)]
    kw = dict(buckets=buckets, batch=args.batch, repeats=args.repeats, seed=0)
    res_dis = measure_drift(model, x_all, disjoint, sched, **kw)
    res_iid = measure_drift(model, x_all, iid, sched, **kw)

    ok = True
    for d, i in zip(res_dis, res_iid):
        print(f"bucket {d['bucket']} | raw {d['zeta2_raw']:.3e} vs {i['zeta2_raw']:.3e}"
              f" | corrected {d['zeta2_corrected']:.3e} vs {i['zeta2_corrected']:.3e}"
              f" | cross {d['zeta2_cross']:.3e} vs {i['zeta2_cross']:.3e}   (disjoint vs iid)")
        ok = ok and d["zeta2_cross"] > 0 and d["zeta2_cross"] > i["zeta2_cross"]
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
