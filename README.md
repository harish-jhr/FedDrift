# Client drift per diffusion timestep

In federated training of a diffusion model, do clients disagree equally at every noise level, or more at some? This repo measures client gradient dissimilarity separately per timestep bucket, for DDPM + FedAvg on CIFAR-10 with Dirichlet label skew.

---

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install the torch build for your GPU first. CIFAR-10 downloads itself to `./data` on first use.

---

## Reproduce

```bash
# 1. client splits (10 clients, alpha 0.1 / 0.5 / 5 / IID), written to ./partitions
python make_partitions.py
python make_partitions.py --seed 1

# 2. check the measurement before trusting anything, should print PASS
python test_drift.py

# 3. the four runs
python -u measure_drift.py --alpha 0.1 > run_alpha0.1.log 2>&1
python -u measure_drift.py --alpha iid > run_iid.log 2>&1
python -u measure_drift.py --alpha 0.1 --seed 1 > run_alpha0.1_seed1.log 2>&1
python -u measure_drift.py --alpha 0.1 --local_epochs 5 --out_dir runs_E5 > run_alpha0.1_E5.log 2>&1

# 4. plots, one figure per measured round
python plot.py --json runs/drift_train0.1_seed0.json
python plot.py --json runs/drift_train0.1_seed0.json --round 10
```

Each run trains FedAvg for 50 rounds and measures drift at rounds 0, 1, 5, 10, 25, 50, for all four splits at the same global weights. Results go to a JSON in `runs/`.

The output name only has the training alpha and the seed in it, so give any other variation its own `--out_dir` or it overwrites.

---

## Files

| File | What it does |
|------|-------------|
| `data.py` | CIFAR-10 as one tensor, Dirichlet / IID client splits |
| `make_partitions.py` | Writes the splits, prints the client x class counts |
| `model.py` | diffusers UNet2DModel, small (9M) and standard (35.7M) |
| `diffusion.py` | DDPM schedule, loss with explicit `t` and `eps`, sampler |
| `fed.py` | Local training and FedAvg |
| `drift.py` | The per-bucket drift measurement |
| `measure_drift.py` | FedAvg run that measures drift along the way |
| `test_drift.py` | Cats-only vs dogs-only clients must show more drift than mixed clients |
| `plot.py` | Six panels per round, one line per split |

---

## How drift is measured

For a fixed global model and one timestep bucket, every client computes its gradient on 256 of its own images. Drift is the average squared distance of the client gradients from their mean.

- all clients are measured at the same weights
- all clients get the same timesteps and the same noise, so only the images differ
- minibatch noise is removed, in two independent ways that are checked against each other

---

## Results so far

Drift divided by within-client noise, for the alpha 0.1 clients at round 50. Buckets go from low noise to high noise.

| Run | 0-200 | 200-400 | 400-600 | 600-800 | 800-1000 |
|-----|:-----:|:-------:|:-------:|:-------:|:--------:|
| alpha 0.1, seed 0 | 1.68 | 8.11 | 13.95 | 12.76 | 12.54 |
| alpha 0.1, seed 1 | 0.75 | 6.12 | 14.40 | 20.49 | 21.70 |
| model trained on IID | 1.31 | 7.52 | 9.27 | 7.75 | 7.43 |
| 5 local epochs | 0.80 | 4.47 | 11.64 | 14.81 | 15.73 |

Same run (seed 0), all four splits:

| Clients | 0-200 | 200-400 | 400-600 | 600-800 | 800-1000 |
|---------|:-----:|:-------:|:-------:|:-------:|:--------:|
| alpha 0.1 | 1.68 | 8.11 | 13.95 | 12.76 | 12.54 |
| alpha 0.5 | 0.70 | 2.98 | 4.96 | 5.52 | 4.07 |
| alpha 5 | 0.07 | 0.21 | 0.27 | 0.55 | 0.28 |
| IID | 0.004 | 0.047 | 0.095 | 0.022 | 0.061 |

What this shows:

- Drift is not uniform over timesteps. Relative to their own noise, clients disagree 7x to 29x more in the mid and high noise buckets than in the lowest one.
- It comes from the label skew. It grows with skew in every bucket and is near zero for IID clients.
- It is learned. On the untrained model the same row is flat (1.6 down to 1.1).
- It repeats across a second seed, a model trained on IID clients, and 5x more local training.

Why it matters: every drift correction method (FedProx, SCAFFOLD) applies one correction strength to the whole loss. If drift sits in particular noise bands, a correction could be aimed there.

Caveats: the models are short runs (about 20k steps, 100k for the 5 epoch one), only two seeds, and the numbers above depend on the measurement batch size (256), so read the shape across buckets and not the absolute values.

---

## To do

- FID, and a longer FedAvg baseline to compare against Tun et al. (arXiv:2311.16538)
- how much FID is lost to skew at all: FedAvg on alpha 0.1 vs on IID, same rounds
- which bucket's drift costs the most: train on alpha 0.1 but give one bucket pooled data
- FedProx and SCAFFOLD baselines, then a timestep-aware correction, compared on rounds to a target FID
- Min-SNR weighted baseline, to check a gain is not just loss reweighting
- third seed, 10 buckets, cross-bucket coherence check
