expdir="./experiments/librispeech_clean100_suffix600_tcpgen200_sche30_drop0.1"
mkdir -p $expdir
python train.py \
    --exp-dir $expdir \
    --librispeech-path /home/gs534/rds/hpc-work/data/Librispeech/ \
    --global-stats-path ./global_stats_100.json \
    --sp-model-path ./spm_unigram_600_100suffix.model \
    --biasing \
    --biasing-list ./blists/rareword_f15.txt \
    --droprate 0.1 \
    --maxsize 200 \
    --epochs 90 \
    # --resume experiments/librispeech_clean100_suffix600_tcpgen500_sche30_nodrop/checkpoints/epoch=45-step=89838.ckpt \
