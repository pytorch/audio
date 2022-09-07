export LD_PRELOAD=/lib64/libgsm.so
expdir="./experiments/librispeech_clean100_suffix_baseline"
ckptpath=$expdir/checkpoints/epoch\=55-step\=109368.ckpt
python eval.py \
    --checkpoint-path $ckptpath \
    --librispeech-path /home/gs534/rds/hpc-work/data/Librispeech/ \
    --sp-model-path ./spm_unigram_1023_100suffix.model \
    --use-cuda
