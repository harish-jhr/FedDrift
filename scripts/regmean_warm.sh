# The best method in the report: RegMean with a FedAvg warm start and a memory term.
# Rounds 1-25 are plain FedAvg (taken from fedavg.sh's round-25 checkpoint), RegMean runs
# from round 26, and the memory term (beta = 0.9) stops repeated corrections piling up.
# Runs uniform and adaptive band strengths side by side.
#
#   bash scripts/fedavg.sh 0 && bash scripts/regmean_warm.sh 0
#   DATASET=mnist ROUNDS=75 REGMEAN_N=512 bash scripts/regmean_warm.sh 0
source "$(dirname "$0")/common.sh"

F=fedavg${SUF}_s$S
[ -f runs/${F}_round25.pt ] || { echo "missing runs/${F}_round25.pt, run scripts/fedavg.sh $S first"; exit 1; }

P="--regmean --regmean_a 0.9 --regmean_n ${REGMEAN_N:-1024} --regmean_beta 0.9 --measure_rounds 50 75 100 --sample_every 10 --resume_round 25 --ref_norm runs/${F}_round{}.pt"
for v in uniform adaptive; do
    n=regmean_warm_${v}${SUF}_s$S
    for f in round25.pt round25_ema.pt; do [ -f runs/${n}_$f ] || cp runs/${F}_$f runs/${n}_$f; done
    [ -f runs/$n.json ] || cp runs/$F.json runs/$n.json
    python -u train.py $A $P --name $n --regmean_rho $v --regmean_warmup 25 > logs/$n.log 2>&1 &
done
wait
for v in uniform adaptive; do
    for r in 50 75 100; do [ $r -le $ROUNDS ] && fid regmean_warm_${v}${SUF}_s$S $r; done
done
