#! /bin/bash

#SBATCH --job-name=torchaudiomodel
#SBATCH --output=/checkpoint/%u/jobs/audio-%A-%a.out
#SBATCH --error=/checkpoint/%u/jobs/audio-%A-%a.err
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --time=1200
#SBATCH --nodes=1
#SBATCH --array=1-32
# number of CPUs = 2x (number of data workers + number of GPUs requested)

>&2 echo $SLURM_JOB_ID

i=$SLURM_ARRAY_TASK_ID

COUNT=1

CMD="srun"
CMD="$CMD python /private/home/vincentqb/audio-pytorch/examples/pipeline_wav2letter/main.py"
# CMD="$CMD --distributed --world-size $SLURM_JOB_NUM_NODES --dist-url 'env://' --dist-backend='nccl'"
#  CMD="$CMD --distributed --world-size $SLURM_JOB_NUM_NODES"
# CMD="$CMD --distributed --world-size 8"
CMD="$CMD --print-freq 1 --reduce-lr-valid --dataset-root /datasets01/librispeech/ --dataset-folder-in-archive 062419"

# choices=(0. 0.2)
# name="dropout"
# l=${#choices[@]}
# j=$(($i % $l))
# i=$(($i / $l))
# item=${choices[$j]}
# CMD="$CMD --$name $item"
# COUNT=$(($COUNT * $l))

# choices=("sum" "mean")
choices=("sum")
name="reduction"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# choices=("mel" "mfcc" "waveform")
# choices=("mfcc" "waveform")
choices=("mfcc")
# choices=("waveform")
name="model-input-type"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# choices=("train-clean-100" "train-clean-100 train-clean-360 train-other-500")
choices=("train-clean-100 train-clean-360 train-other-500")
name="dataset-train"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

choices=("dev-clean")
name="dataset-valid"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

choices=(128)
name="batch-size"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

choices=(.6)
name="learning-rate"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# momentums=(0. .8)
choices=(.8)
name="momentum"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

choices=(.00001)
name="weight-decay"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# grads=(0. .2)
# choices=(0.)
# name="clip-grad"
# l=${#choices[@]}
# j=$(($i % $l))
# i=$(($i / $l))
# item=${choices[$j]}
# CMD="$CMD --$name $item"
# COUNT=$(($COUNT * $l))

# gammas=(.98 .99)
choices=(.99)
name="gamma"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# choices=(80 160)
choices=(160)
name="hop-length"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# choices=(2000 1000)
# name="hidden-channels"
# l=${#choices[@]}
# j=$(($i % $l))
# i=$(($i / $l))
# item=${choices[$j]}
# CMD="$CMD --$name $item"
# COUNT=$(($COUNT * $l))

# choices=(512 400)
choices=(400)
name="win-length"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# nbinss=(13 128 40)
choices=(13)
name="bins"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

choices=("--normalize")
# choices=("")
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD $item"
COUNT=$(($COUNT * $l))

# choices=("" "--time-mask 70 --freq-mask 7" "--time-mask 35 --freq-mask 5")
choices=("")
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD $item"
COUNT=$(($COUNT * $l))

# choices=(0 35)
# # choices=(0)
# name="time-mask"
# l=${#choices[@]}
# j=$(($i % $l))
# i=$(($i / $l))
# item=${choices[$j]}
# CMD="$CMD --$name $item"
# COUNT=$(($COUNT * $l))
# 
# choices=(0 5)
# # choices=(0)
# name="freq-mask"
# l=${#choices[@]}
# j=$(($i % $l))
# i=$(($i / $l))
# item=${choices[$j]}
# CMD="$CMD --$name $item"
# COUNT=$(($COUNT * $l))

# choices=("sgd" "adadelta" "adamw")
# choices=("sgd" "adadelta")
choices=("adadelta")
name="optimizer"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# schedulers=("exponential" "reduceonplateau")
choices=("reduceonplateau")
name="scheduler"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# decoders=("greedy" "greedyiter" "viterbi")
choices=("greedy")
name="decoder"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

# decoders=("greedy" "greedyiter" "viterbi")
choices=(1000)
name="max-epoch"
l=${#choices[@]}
j=$(($i % $l))
i=$(($i / $l))
item=${choices[$j]}
CMD="$CMD --$name $item"
COUNT=$(($COUNT * $l))

if [[ "$SLURM_ARRAY_TASK_COUNT" -ne $COUNT ]]; then
    >&2 echo "SLURM_ARRAY_TASK_COUNT = $SLURM_ARRAY_TASK_COUNT is not equal to $COUNT"
    exit
fi

# The ENV below are only used in distributed training with env:// initialization
# export MASTER_ADDR=${SLURM_JOB_NODELIST:0:9}${SLURM_JOB_NODELIST:10:4}
# export MASTER_PORT=29500

# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

HASH=`echo "$CMD $SLURM_JOB_ID" | md5sum | awk '{print $1}'`

CMD="$CMD --checkpoint /checkpoint/vincentqb/checkpoint/checkpoint-$SLURM_JOB_ID-$HASH.pth.tar"

>&2 echo $CMD
eval $CMD
