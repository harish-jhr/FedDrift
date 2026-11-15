# Reference experiment that is NOT data-free: every client gets the same 1000 test images
# and adds 13 of them to each batch, used either at all timesteps or only at t < 400.
# SHARE_SOURCE=cifar100 uses CIFAR-100 images instead (no class overlap with CIFAR-10).
source "$(dirname "$0")/common.sh"

SRC=${SHARE_SOURCE:-cifar10_test}
TAG=$([ "$SRC" = cifar100 ] && echo _c100 || echo "")
P="--share_size 1000 --share_source $SRC --share_batch 13 --measure_rounds 25 50 100"
python -u train.py $A $P --name shared${TAG}_all_t${SUF}_s$S --share_range 0 1000 > logs/shared${TAG}_all_t${SUF}_s$S.log 2>&1 &
python -u train.py $A $P --name shared${TAG}_low_t${SUF}_s$S --share_range 0 400 > logs/shared${TAG}_low_t${SUF}_s$S.log 2>&1 &
wait
for n in shared${TAG}_all_t${SUF}_s$S shared${TAG}_low_t${SUF}_s$S; do
    for r in 25 50 100; do fid $n $r; done
done
