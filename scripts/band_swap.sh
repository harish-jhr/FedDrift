# Diagnostic: a model trained on skewed clients (Dirichlet 0.1) generates images, but one
# timestep band of the reverse process is handed to a model trained on IID clients, and
# the other way round. The change in FID shows which bands the skew actually hurts.
# Trains both models for 200 rounds first if their checkpoints are missing.
source "$(dirname "$0")/common.sh"

R=200
python make_partitions.py --dataset "$DATASET" --alphas 0.1 iid --seed "$S" > /dev/null
for a in 0.1 iid; do
    n=swapbase_a${a}${SUF}_s$S
    [ -f runs/${n}_round$R.pt ] || python -u train.py $D --alpha $a --measure_alphas $a --rounds $R --seed $S --out_dir runs --name $n --measure_rounds 100 $R > logs/$n.log 2>&1 &
done
wait

SKEW=runs/swapbase_a0.1${SUF}_s${S}_round$R.pt
IID=runs/swapbase_aiid${SUF}_s${S}_round$R.pt
python fid.py $D --ckpt $SKEW --log fid.txt
python fid.py $D --ckpt $IID --log fid.txt
for band in "0 200" "200 400" "400 600" "600 800" "800 1000"; do
    python fid.py $D --ckpt $SKEW --swap_ckpt $IID --swap_range $band --log fid.txt
    python fid.py $D --ckpt $IID --swap_ckpt $SKEW --swap_range $band --log fid.txt
done
