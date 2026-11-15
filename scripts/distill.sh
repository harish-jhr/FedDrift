# Distillation: every 10 rounds the server generates 1000 images with the global model,
# and clients match the global model's predictions on them (32 per batch).
# Uniform strength vs strength per timestep band proportional to the measured drift.
source "$(dirname "$0")/common.sh"

P="--distill 1.0 --distill_n 1000 --distill_every 10 --distill_batch 32 --distill_start 10 --prox_ramp 20"
python -u train.py $A $P --name distill_uniform${SUF}_s$S --measure_rounds 25 50 100 > logs/distill_uniform${SUF}_s$S.log 2>&1 &
python -u train.py $A $P --name distill_adaptive${SUF}_s$S --prox_adapt --measure_rounds 10 20 25 30 40 50 60 70 80 90 100 > logs/distill_adaptive${SUF}_s$S.log 2>&1 &
wait
for n in distill_uniform${SUF}_s$S distill_adaptive${SUF}_s$S; do
    for r in 25 50 100; do fid $n $r; done
done
