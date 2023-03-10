# Speech Recognition Inference with CUDA CTC Beam Search Decoder

This is an example inference script for running decoding on the LibriSpeech dataset and [zipformer](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_ctc) models, using a CUDA-based CTC beam search decoder that supports parallel decoding through batch and vocabulary axises.

## Usage
Additional command line parameters and information can is available with the `--help` option.

Sample command

```
pip install sentencepiece

# download pretrained files
wget -nc https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01/resolve/main/data/lang_bpe_500/bpe.model
wget -nc https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01/resolve/main/exp/cpu_jit.pt

python inference.py \
    --librispeech_path ./librispeech/ \
    --split test-other \
    --model ./cpu_jit.pt \
    --bp-model ./bpe.model \
    --beam-size 10 \
    --blank-skip-threshold 0.95
```

## Results
The table below contains throughput and WER benchmark results on librispeech test_other set between cuda ctc decoder and flashlight cpu decoder.

(Note: batch_size=4, beam_size=10, nbest=10, vocab_size=500, no LM, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, V100 GPU)

| Decoder | Setting | WER (%) | N-Best Oracle WER (%) | Decoder Cost Time (seconds) |
|:-----------|-----------:|-----------:|-----------:|-----------:|
|CUDA decoder|blank_skip_threshold=0.95| 5.81 | 4.11 | 2.57 |
|CUDA decoder|blank_skip_threshold=1.0 (no frame-skip)| 5.81 | 4.09 | 6.24 |
|flashlight decoder|beam_size_token=10| 5.86 | 4.30 | 28.61 |
|flashlight decoder|beam_size_token=vocab_size| 5.86 | 4.30 | 791.80 |

