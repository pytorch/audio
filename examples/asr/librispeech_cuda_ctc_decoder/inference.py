import argparse
import logging
import time

import sentencepiece as spm
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchaudio.models.decoder import ctc_decoder, cuda_ctc_decoder

logger = logging.getLogger(__name__)


def collate_wrapper(batch):
    speeches, labels = [], []
    for (speech, _, label, _, _, _) in batch:
        speeches.append(speech)
        labels.append(label.strip().lower().strip())
    return speeches, labels


def run_inference(args):
    device = torch.device("cuda", 0)
    model = torch.jit.load(args.model)
    model.to(device)
    model.eval()

    bpe_model = spm.SentencePieceProcessor()
    bpe_model.load(args.bpe_model)
    vocabs = [bpe_model.id_to_piece(id) for id in range(bpe_model.get_piece_size())]
    if args.using_cpu_decoder:
        cpu_decoder = ctc_decoder(
            lexicon=None,
            tokens=vocabs,
            lm=None,
            nbest=args.nbest,
            beam_size=args.beam_size,
            beam_size_token=args.beam_size_token,
            beam_threshold=args.beam_threshold,
            blank_token="<blk>",
            sil_token="<blk>",
        )
    else:
        assert vocabs[0] == "<blk>", "idx of blank token has to be zero"
        blank_frame_skip_threshold = float(torch.log(torch.tensor(args.blank_skip_threshold)))
        cuda_decoder = cuda_ctc_decoder(
            vocabs, nbest=args.nbest, beam_size=args.beam_size, blank_skip_threshold=blank_frame_skip_threshold
        )

    dataset = torchaudio.datasets.LIBRISPEECH(args.librispeech_path, url=args.split, download=True)

    total_edit_distance, oracle_edit_distance, total_length = 0, 0, 0

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, collate_fn=collate_wrapper
    )

    decoding_duration = 0
    for idx, batch in enumerate(data_loader):
        waveforms, transcripts = batch
        waveforms = [wave.to(device) for wave in waveforms]
        features = [torchaudio.compliance.kaldi.fbank(wave, num_mel_bins=80, snip_edges=False) for wave in waveforms]
        feature_lengths = [f.size(0) for f in features]

        features = pad_sequence(features, batch_first=True, padding_value=torch.log(torch.tensor(1e-10)))
        feature_lengths = torch.tensor(feature_lengths, device=device)

        encoder_out, encoder_out_lens = model.encoder(
            x=features,
            x_lens=feature_lengths,
        )
        nnet_output = model.ctc_output(encoder_out)
        log_prob = torch.nn.functional.log_softmax(nnet_output, -1)

        decoding_start = time.perf_counter()
        preds = []
        if args.using_cpu_decoder:
            results = cpu_decoder(log_prob.cpu())
            duration = time.perf_counter() - decoding_start
            for i in range(len(results)):
                ith_preds = bpe_model.decode([results[i][j].tokens.tolist() for j in range(len(results[i]))])
                ith_preds = [pred.lower().split() for pred in ith_preds]
                preds.append(ith_preds)
        else:
            results = cuda_decoder(log_prob, encoder_out_lens.to(torch.int32))
            duration = time.perf_counter() - decoding_start
            for i in range(len(results)):
                ith_preds = bpe_model.decode([results[i][j].tokens for j in range(len(results[i]))])
                ith_preds = [pred.lower().split() for pred in ith_preds]
                preds.append(ith_preds)
        decoding_duration += duration

        for transcript, nbest_pred in zip(transcripts, preds):
            total_edit_distance += torchaudio.functional.edit_distance(transcript.split(), nbest_pred[0])
            oracle_edit_distance += min(
                [torchaudio.functional.edit_distance(transcript.split(), nbest_pred[i]) for i in range(len(nbest_pred))]
            )
            total_length += len(transcript.split())

        if idx % 10 == 0:
            logger.info(
                f"Processed elem {idx}; "
                f"WER: {total_edit_distance / total_length}, "
                f"Oracle WER: {oracle_edit_distance / total_length}, ",
                f"decoding time for batch size {args.batch_size}: {duration}",
            )

    logger.info(
        f"Final WER: {total_edit_distance / total_length}, ",
        f"Oracle WER: {oracle_edit_distance / total_length}, ",
        f"time for decoding {decoding_duration} [sec].",
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--librispeech_path",
        type=str,
        help="folder where LibriSpeech is stored",
        default="./librispeech",
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
        default="./cpu_jit.pt",
        help="pretrained ASR model using CTC loss",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="./bpe.model",
        help="bpe file for pretrained ASR model",
    )
    parser.add_argument(
        "--nbest",
        type=int,
        default=10,
        help="number of best hypotheses to return",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="beam size for determining number of hypotheses to store",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size for decoding",
    )
    parser.add_argument(
        "--blank-skip-threshold",
        type=float,
        default=0.95,
        help="skip frames where prob_blank > 0.95, https://ieeexplore.ieee.org/document/7736093",
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    # cpu decoder specific parameters
    parser.add_argument("--using-cpu-decoder", action="store_true", help="whether to use flashlight cpu ctc decoder")
    parser.add_argument("--beam-threshold", type=int, default=50, help="beam threshold for pruning hypotheses")
    parser.add_argument(
        "--beam-size-token",
        type=int,
        default=None,
        help="number of tokens to consider at each beam search step",
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
