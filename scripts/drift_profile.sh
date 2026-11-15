# Client drift per timestep band for every client split, measured on one FedAvg run
# (trained on Dirichlet 0.1 by default) for 200 rounds. Plot it with:
#   python plots/drift_curves.py --json runs/drift_profile_s0.json
source "$(dirname "$0")/common.sh"

TRAIN=${TRAIN_ALPHA:-0.1}
python make_partitions.py --dataset "$DATASET" --alphas 0.1 0.5 5 iid class --seed "$S" > /dev/null
n=drift_profile${SUF}_s$S
python -u train.py $D --alpha $TRAIN --measure_alphas 0.1 0.5 5 iid class --rounds 200 --seed $S --out_dir runs \
    --name $n --measure_rounds 0 1 5 10 25 50 100 150 200 > logs/$n.log 2>&1
