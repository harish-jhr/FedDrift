# Short tests behind the report's one-round comparison (Figure "one round from round 100").
# Each test starts from FedAvg's round-100 checkpoint, runs one federated round with a
# different way of combining the clients, and scores it with FID on 2,000 samples.
# 2k FID reads about 11 points higher than 10k FID, so compare these only with each other.
#
#   bash scripts/fedavg.sh 0 && bash scripts/one_round_tests.sh 0
#   TESTS="regmean fisher" bash scripts/one_round_tests.sh 0
source "$(dirname "$0")/common.sh"

C=runs/fedavg_s${S}_round100.pt
[ -f $C ] || { echo "missing $C, run scripts/fedavg.sh $S first"; exit 1; }
O=runs/one_round
mkdir -p $O

for test in ${TESTS:-regmean band_strengths fisher self_replay correction_before_adam curvature ten_rounds}; do
    case $test in
        regmean)                cmd="experiments/regmean_one_round.py --ckpt $C --out $O/regmean" ;;
        band_strengths)         cmd="experiments/band_strength_tests.py --ckpt $C --out $O/band_strengths" ;;
        fisher)                 cmd="experiments/fisher_merge.py --ckpt $C --out $O/fisher" ;;
        self_replay)            cmd="experiments/self_replay.py --ckpt $C --out $O/self_replay" ;;
        correction_before_adam) cmd="experiments/correction_before_adam.py --ckpt $C --out $O/correction_before_adam" ;;
        curvature)              cmd="experiments/curvature_prediction.py --ckpt $C --out $O/curvature" ;;
        ten_rounds)             cmd="experiments/regmean_ten_rounds.py --ckpt runs/fedavg_s${S}_round25.pt --ema runs/fedavg_s${S}_round25_ema.pt --out $O/regmean_ten_rounds" ;;
        *) echo "unknown test $test"; continue ;;
    esac
    python -u $cmd > logs/one_round_$test.log 2>&1
done
