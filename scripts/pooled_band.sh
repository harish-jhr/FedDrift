# Diagnostic, NOT data-free: clients train normally, except that samples whose timestep
# falls in one band use images from the whole dataset. This removes client disagreement
# in that band, and the FID recovered shows what that band's disagreement was costing.
source "$(dirname "$0")/common.sh"

for band in "0 400" "400 800"; do
    n=pool_${band// /-}${SUF}_s$S
    python -u train.py $A --name $n --pool_range $band --measure_rounds 50 100 > logs/$n.log 2>&1 &
done
wait
for band in "0 400" "400 800"; do
    n=pool_${band// /-}${SUF}_s$S
    for r in 50 100; do fid $n $r; done
done
