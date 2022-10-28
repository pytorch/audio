. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate torchaudio
export LD_PRELOAD=/lib64/libgsm.so
# expdir="./experiments/librispeech_clean100_suffix_baseline"
expdir="./experiments/librispeech_clean100_suffix_attndeepbiasing"
decode_dir=$expdir/decode_test_clean_b10
mkdir -p $decode_dir
ckptpath=$expdir/checkpoints/epoch\=69-step\=136710.ckpt
# ckptpath=$expdir/checkpoints/epoch\=78-step\=154287.ckpt
python eval.py \
    --checkpoint-path $ckptpath \
    --librispeech-path /home/gs534/rds/hpc-work/data/Librispeech/ \
    --sp-model-path ./spm_unigram_1023_100suffix.model \
    --expdir $decode_dir \
    --use-cuda \
    --biasinglist ./blists/rareword_f30.txt \
    --droprate 0.0 \
    --maxsize 1000 \
