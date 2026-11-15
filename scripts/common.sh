# Settings shared by every script in scripts/. Each script takes the seed as its only
# argument, and the dataset and training objective come from environment variables:
#
#   bash scripts/fedavg.sh 0
#   DATASET=mnist ROUNDS=75 bash scripts/fedavg.sh 0
#   OBJECTIVE=fm bash scripts/regmean.sh 1
#
# DATASET:   cifar10 (default), mnist, celeba64, tinyimagenet
# OBJECTIVE: ddpm (default) or fm (flow matching)
# ALPHA:     client split; class (one class per client) by default, 0.1 for celeba64
# ROUNDS:    communication rounds, 100 by default

cd "$(dirname "$0")/.."
S=${1:-0}
DATASET=${DATASET:-cifar10}
OBJECTIVE=${OBJECTIVE:-ddpm}
if [ "$DATASET" = celeba64 ]; then ALPHA=${ALPHA:-0.1}; else ALPHA=${ALPHA:-class}; fi
ROUNDS=${ROUNDS:-100}

# run names get a suffix only when they differ from the main CIFAR-10 / DDPM / one-class setup
SUF=""
[ "$DATASET" != cifar10 ] && SUF="${SUF}_$DATASET"
[ "$OBJECTIVE" != ddpm ] && SUF="${SUF}_$OBJECTIVE"
[ "$ALPHA" != class ] && SUF="${SUF}_a$ALPHA"

D="--dataset $DATASET --objective $OBJECTIVE"
A="$D --alpha $ALPHA --measure_alphas $ALPHA --rounds $ROUNDS --seed $S --out_dir runs"
mkdir -p logs runs
python make_partitions.py --dataset "$DATASET" --alphas "$ALPHA" --seed "$S" > /dev/null

fid() {
    python fid.py $D --ckpt "runs/$1_round$2_ema.pt" --log fid.txt
}
