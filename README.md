# FedDrift: does client drift in federated diffusion depend on the noise level?

When a diffusion model is trained with federated learning, every client trains a copy of the model on its own private images, and a server combines the copies each round. If the clients hold different kinds of images, their copies pull in different directions. This is called client drift, and it makes the combined model noticeably worse than one trained on all the data together.

A diffusion model is trained on many noise levels at once. This project asks whether clients disagree equally at every noise level, and whether a fix should treat noise levels differently. It contains the code for every experiment in our course report: the drift measurements, the diagnostic experiments, the baselines, and the method that worked best.

## What we found

- **Drift depends strongly on the noise level, and it moves during training.** On CIFAR-10 with one class per client, clients disagree most at low noise at the start, and by round 25 most of the disagreement sits in the middle noise range.
- **The noise range with the most drift is not the one that hurts image quality most.** Removing client disagreement below timestep 400 recovers 72% of the quality gap to IID training, while removing it in the high-drift middle range recovers only 27%.
- **Pulling clients towards each other does not help.** FedProx and distillation, even with a separate strength for each noise range, end level with FedAvg or worse. Sharing a few real images helps a lot, which tells us that clients are missing information about the classes they do not have, not agreement.
- **RegMean merging brings that information in without sharing images.** Instead of averaging the client models, the server combines them using summary statistics (Gram matrices) of each client's data. With a strength per noise range, a 25-round FedAvg warm start and a memory term that keeps it stable, it beats FedAvg on CIFAR-10 and MNIST.
- **With flow matching instead of DDPM, the drift moves to the noisiest range.** Where clients disagree depends on what the model is trained to predict.

## Main results

CIFAR-10, ten clients with one class each, FID on 10,000 samples from the EMA model (lower is better). "Gap closed" is how much of the distance between FedAvg and IID training (FID 23.51) a method recovers.

| Method | Round 25 | Round 50 | Round 75 | Round 100 | Gap closed |
|---|:--:|:--:|:--:|:--:|:--:|
| FedAvg | 141.2 | 52.0 | 45.0 | 43.65 | 0% |
| FedProx, uniform | 131.7 | 52.1 | – | 43.92 | −1% |
| FedProx, drift-adaptive | 112.4 | 53.3 | – | 43.84 | −1% |
| Distillation, uniform | 143.0 | 75.8 | – | 51.90 | −41% |
| Distillation, drift-adaptive | 125.4 | 74.8 | – | 57.80 | −69% |
| RegMean, uniform, from scratch | 134.4 | 52.6 | – | 43.47 | 1% |
| RegMean, adaptive, from scratch | 172.8 | 50.8 | – | 52.65 | −45% |
| RegMean, uniform, warm start + memory | – | 56.0 | 44.2 | 42.54 | 6% |
| **RegMean, adaptive, warm start + memory** | – | **51.8** | **43.0** | **41.27** | **12%** |
| *Shared images, all timesteps (not data-free)* | 116.5 | 45.5 | – | 31.88 | 58% |
| *Shared images, timesteps below 400 (not data-free)* | 136.0 | 47.9 | – | 28.75 | 74% |

The best data-free method reaches FedAvg's round-100 quality by round 75.

On **MNIST** (one digit per client, 75 rounds), RegMean with warm start and memory scores 8.09 against FedAvg's 8.88 at round 50, and 7.02 against 7.28 at round 75.

With **flow matching** on CIFAR-10 (mean of two seeds, RegMean from scratch), uniform RegMean reaches 39.9 and adaptive RegMean 41.3 at round 100, against 43.6 for FedAvg.

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install the PyTorch build that matches your GPU first. CIFAR-10 and MNIST download themselves into `./data` the first time they are used. CelebA-64 and Tiny-ImageNet are downloaded from Hugging Face and cached in `./data` as `.npz` files. Experiments are logged to Weights & Biases; add `--no_wandb` to `train.py`, or set `WANDB_MODE=offline`, to turn that off.

## Running the experiments

Every experiment has a script in `scripts/`. Each one takes the seed as its only argument, trains the models, and then computes FID at the reported rounds. Run them from anywhere:

```bash
bash scripts/fedavg.sh 0
```

The dataset, training objective and number of rounds are set with environment variables, so the same scripts cover every setting:

```bash
DATASET=mnist ROUNDS=75 bash scripts/fedavg.sh 0
OBJECTIVE=fm bash scripts/regmean.sh 0
```

| Script | What it runs |
|---|---|
| `fedavg.sh` | FedAvg baseline. Also measures drift per noise range and saves the round-25 checkpoint used by the warm start. |
| `regmean_warm.sh` | The best method: RegMean with a FedAvg warm start and memory, uniform and adaptive strengths. Run `fedavg.sh` first. |
| `regmean.sh` | RegMean from scratch, uniform and adaptive strengths. |
| `fedprox.sh` | FedProx on the model's predictions, uniform and drift-adaptive strength. |
| `distill.sh` | Distillation on images generated by the global model, uniform and drift-adaptive strength. |
| `shared_data.sh` | Reference with 13 shared images per batch, at all timesteps or only below 400. `SHARE_SOURCE=cifar100` uses CIFAR-100 images instead. |
| `pooled_band.sh` | Diagnostic: clients use images from the whole dataset in one noise range (0–400 or 400–800). |
| `band_swap.sh` | Diagnostic: a skew-trained model hands one noise range of generation to an IID-trained model, and the other way round. |
| `drift_profile.sh` | Drift in every noise range for every client split (IID, Dirichlet 5, 0.5, 0.1, one class), over 200 rounds. |
| `one_round_tests.sh` | The short one-round tests from FedAvg's round-100 checkpoint: RegMean and its controls, Fisher merging, self-replay, correction before Adam, and others. See `experiments/README.md`. |

To reproduce the MNIST result:

```bash
DATASET=mnist ROUNDS=75 bash scripts/fedavg.sh 0
DATASET=mnist ROUNDS=75 REGMEAN_N=512 bash scripts/regmean_warm.sh 0
```

Checkpoints and drift measurements go to `runs/`, training logs to `logs/`, and every FID score is appended to `fid.txt`. `scripts/skypilot.yaml` launches several scripts in parallel on a Kubernetes GPU cluster, if you have one.

## How the code is organised

| File | What it does |
|---|---|
| `train.py` | The training script behind every experiment: local training, merging, drift measurement, checkpoints. `python train.py --help` lists all options. |
| `regmean.py` | Gram matrices per noise range, the band-weighted RegMean merge, the memory term and the adaptive strengths. |
| `fed.py` | Local training on one client (with the optional FedProx, distillation, shared-image and pooled-band changes) and FedAvg. |
| `drift.py` | Measures how much client gradients disagree in each noise range, with mini-batch noise removed. |
| `diffusion.py` | DDPM and flow-matching noise schedules, the loss, and the samplers. |
| `model.py` | The U-Net: 9.0M parameters at 32×32, 6.5M for MNIST at 28×28, 9.6M at 64×64. |
| `data.py` | Datasets and client splits (one class per client, Dirichlet, IID). |
| `make_partitions.py` | Writes the client splits to `partitions/`. The scripts call it automatically. |
| `fid.py` | FID of a checkpoint against the training set, including band-swap models. |
| `sample.py` | Draws sample grids from a checkpoint. |
| `experiments/` | The short tests and diagnostics: one-round merges, client output disagreement, denoising error per band, memorisation check. Each script is described in `experiments/README.md`. |
| `plots/` | `report_figures.py` and `sketch_figure.py` redraw the report's figures; `drift_curves.py` plots drift from a run's JSON file. |
| `tests/test_drift.py` | Checks the drift measurement: two clients with different classes must disagree more than two mixed clients. |

## How the main method works

After local training, each client runs noisy copies of its own images through its model and records, for every layer, the Gram matrix of that layer's inputs, separately for each of five noise ranges. The server then picks, for each layer, the weights whose outputs best match every client's outputs on that client's own inputs. Where all clients' Gram matrices agree this is plain averaging; where they differ, each client gets more say in the directions its own data uses.

The merged update splits exactly into the FedAvg update plus one correction per noise range, and each correction gets its own strength. With adaptive strengths, every ten rounds each client reports how much its local training improved the loss on held-out images in each range, and ranges where that improvement does not carry over get a weaker correction. The first 25 rounds are plain FedAvg, and a memory term removes the part of the correction that repeats every round, which keeps the weights from growing. Clients send their model, one Gram matrix per layer and range, and five numbers. No images leave a client.

## Team

Harishwar Rao J., Anmol Agarwal and Aditi Mishra, IISER Bhopal. Course project for ECS431 (Federated Learning), 2026.
