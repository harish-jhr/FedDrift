# FedAvg baseline. Also measures client drift per timestep band and saves the
# round-25 checkpoint that regmean_warm.sh starts from.
source "$(dirname "$0")/common.sh"

n=fedavg${SUF}_s$S
python -u train.py $A --name $n --measure_rounds 10 25 50 75 100 > logs/$n.log 2>&1
for r in 25 50 75 100; do
    [ $r -le $ROUNDS ] && fid $n $r
done
