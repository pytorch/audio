. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate torchaudio
export LD_PRELOAD=/lib64/libgsm.so
expdir="./experiments/librispeech_clean100_suffix_baseline"
mkdir -p $expdir
python train.py \
    --exp-dir $expdir \
    --librispeech-path /home/gs534/rds/hpc-work/data/Librispeech/ \
    --global-stats-path ./global_stats_100.json \
    --sp-model-path ./spm_unigram_1023_100suffix.model \
    --biasinglist ./blists/rareword_f30.txt \
    --droprate 0.3 \
    --maxsize 1000 \
    --epochs 160 \
