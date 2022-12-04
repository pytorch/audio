. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate cuda113
export LD_PRELOAD=/lib64/libgsm.so
expdir="./experiments/librispeech_clean100_suffix600_tcpgen500_sche30_nodrop"
mkdir -p $expdir
python train.py \
    --exp-dir $expdir \
    --librispeech-path /home/gs534/rds/hpc-work/data/Librispeech/ \
    --global-stats-path ./global_stats_100.json \
    --sp-model-path ./spm_unigram_600_100suffix.model \
    --biasing true \
    --biasinglist ./blists/rareword_f15.txt \
    --droprate 0.0 \
    --maxsize 500 \
    --epochs 90 \
    --resume experiments/librispeech_clean100_suffix600_tcpgen500_sche30_nodrop/checkpoints/epoch=45-step=89838.ckpt \
