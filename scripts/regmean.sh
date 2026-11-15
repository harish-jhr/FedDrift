# RegMean aggregation from scratch (every round, no warm start, no memory),
# with uniform band strengths and with adaptive band strengths.
# REGMEAN_N sets the noisy images per band for the Gram matrices (default 1024).
source "$(dirname "$0")/common.sh"

P="--regmean --regmean_a 0.9 --regmean_n ${REGMEAN_N:-1024} --measure_rounds 25 50 75 100 --sample_every 10"
python -u train.py $A $P --name regmean_uniform${SUF}_s$S --regmean_rho uniform > logs/regmean_uniform${SUF}_s$S.log 2>&1 &
python -u train.py $A $P --name regmean_adaptive${SUF}_s$S --regmean_rho adaptive > logs/regmean_adaptive${SUF}_s$S.log 2>&1 &
wait
for n in regmean_uniform${SUF}_s$S regmean_adaptive${SUF}_s$S; do
    for r in 25 50 75 100; do [ $r -le $ROUNDS ] && fid $n $r; done
done
