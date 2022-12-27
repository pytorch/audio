import argparse
import logging
import time
from typing import Optional

import torch
import torchaudio
import torch._dynamo.config
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files


logger = logging.getLogger(__name__)


class CollateFn:
    """Custom collation function to pad every waveform to the max size."""
    def __init__(self, max_waveform_len=-1):
        self.max_waveform_len = max_waveform_len

    def __call__(self, batch):
        waveforms = [item[0] for item in batch]
        transcripts = [item[2] for item in batch]
        lengths = torch.tensor([waveform.shape[1] for waveform in waveforms])
        waveforms = torch.nn.utils.rnn.pad_sequence([w[0] for w in waveforms], batch_first=True)
        # Pad the batched waveforms to the max length
        cur_length = waveforms.shape[1]
        if cur_length < self.max_waveform_len:
            extra_padding = torch.zeros((len(batch), self.max_waveform_len - cur_length))
            waveforms = torch.hstack((waveforms, extra_padding))
        return waveforms, lengths, transcripts


def run_inference(args):
    # get pretrained wav2vec2.0 model
    bundle = getattr(torchaudio.pipelines, args.model)
    model = bundle.get_model()

    if args.use_cuda:
        gpu_device = torch.device("cuda")
        cpu_device = torch.device("cpu")
        model = model.to(gpu_device)

    if args.compile:
        # torch._dynamo.config.verbose=True
        # Ideally, you'd want to compile with model with dynamic tensor sizing enabled, but this currently
        # breaks.
        # model = torch.compile(model, dynamic=True)
        model = torch.compile(model)

    # get decoder files
    files = download_pretrained_files("librispeech-4-gram")

    decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        nbest=args.nbest,
        beam_size=args.beam_size,
        beam_size_token=args.beam_size_token,
        beam_threshold=args.beam_threshold,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        unk_score=args.unk_score,
        sil_score=args.sil_score,
        log_add=False,
    )

    dataset = torchaudio.datasets.LIBRISPEECH(args.librispeech_path, url=args.split, download=False)
    if args.batch_size > 1:
        collate_fn = CollateFn(max_waveform_len=args.max_waveform_length)
    else:
        collate_fn = CollateFn(max_waveform_len=-1)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    total_edit_distance = 0
    total_length = 0
    num_samples = 0
    num_model_evals = 0
    for sample in dataset:
        waveforms, lengths, transcripts = sample
        if args.use_cuda:
            waveforms = waveforms.to(gpu_device)
            lengths = lengths.to(gpu_device)

        NUM_RUNS = 20
        total_time = 0.0
        for i in range(NUM_RUNS):
            start_time = time.time()
            # inference_mode() doesn't seem to work with compile() yet, at least for this model.
            with torch.no_grad():
                # We should be passing in 'lengths' here, but that doesn't work with compile() yet.
                # WARNING: This means that the results will not be correct!
                # emissions, emission_lengths = model(waveforms, lengths)
                emissions, emission_lengths = model(waveforms)
            elapsed_time = time.time() - start_time
            print("Model evaluation %d took %f s" % (num_model_evals, elapsed_time))
            if i > 0:  # Don't include the first run in timings.
                total_time += elapsed_time
        print('Average runtime = %f' % (total_time / (NUM_RUNS - 1)))

        if args.use_cuda:
            emissions = emissions.to(cpu_device)
            emission_lengths = emission_lengths.to(cpu_device)

        for i in range(len(transcripts)):
            # Due to the removal of 'lengths' above, emission_lengths will be None and the following
            # line will break.
            emission = emissions[i:i + 1, 0:emission_lengths[i], :]
            result = decoder(emission)
            transcript = transcripts[i].strip().lower().strip()
            total_edit_distance += torchaudio.functional.edit_distance(transcript.split(), result[0][0].words)
            total_length += len(transcript.split())

        num_samples += len(transcripts)
        logger.info(f"Processed elem {num_samples}; WER: {total_edit_distance / total_length}")
    logger.info(f"Final WER: {total_edit_distance / total_length}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--librispeech_path",
        type=str,
        help="folder where LibriSpeech is stored",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="LibriSpeech dataset split",
        choices=["dev-clean", "dev-other", "test-clean", "test-other"],
        default="test-other",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="WAV2VEC2_ASR_BASE_960H",
        help="pretrained Wav2Vec2 model from torchaudio.pipelines",
    )
    parser.add_argument("--nbest", type=int, default=1, help="number of best hypotheses to return")
    parser.add_argument(
        "--beam-size", type=int, default=500, help="beam size for determining number of hypotheses to store"
    )
    parser.add_argument(
        "--beam-size-token",
        type=Optional[int],
        default=None,
        help="number of tokens to consider at each beam search step",
    )
    parser.add_argument("--beam-threshold", type=int, default=50, help="beam threshold for pruning hypotheses")
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=1.74,
        help="languge model weight",
    )
    parser.add_argument(
        "--word-score",
        type=float,
        default=0.52,
        help="word insertion score",
    )
    parser.add_argument("--unk_score", type=float, default=float("-inf"), help="unknown word insertion score")
    parser.add_argument("--sil_score", type=float, default=0, help="silence insertion score")
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Run using CUDA.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use PyTorch 2.0 compile optimizations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for running inference."
    )
    parser.add_argument(
        "--max_waveform_length",
        type=int,
        default=552160,
        help="Max waveform length to use when batching."
    )
    return parser.parse_args()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def _main():
    args = _parse_args()
    _init_logger(args.debug)
    run_inference(args)


if __name__ == "__main__":
    _main()
