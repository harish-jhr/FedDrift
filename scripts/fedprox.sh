# Function-space FedProx: each client's predictions are pulled towards the global model's.
# Uniform strength vs strength per timestep band proportional to the measured drift.
source "$(dirname "$0")/common.sh"

P="--prox 0.4 --prox_ramp 20"
python -u train.py $A $P --name prox_uniform${SUF}_s$S --measure_rounds 25 50 100 > logs/prox_uniform${SUF}_s$S.log 2>&1 &
python -u train.py $A $P --name prox_adaptive${SUF}_s$S --prox_adapt --measure_rounds 10 20 25 30 40 50 60 70 80 90 100 > logs/prox_adaptive${SUF}_s$S.log 2>&1 &
wait
for n in prox_uniform${SUF}_s$S prox_adaptive${SUF}_s$S; do
    for r in 25 50 100; do fid $n $r; done
done
