import argparse
import logging
import os
import unittest

from interactive_asr.utils import setup_asr, transcribe_file


class ASRTest(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    arguments_dict = {
        "path": "/scratch/jamarshon/downloads/model.pt",
        "input_file": "/scratch/jamarshon/audio/examples/interactive_asr/data/sample.wav",
        "data": "/scratch/jamarshon/downloads",
        "user_dir": "/scratch/jamarshon/fairseq-py/examples/speech_recognition",
        "no_progress_bar": False,
        "log_interval": 1000,
        "log_format": None,
        "tensorboard_logdir": "",
        "tbmf_wrapper": False,
        "seed": 1,
        "cpu": True,
        "fp16": False,
        "memory_efficient_fp16": False,
        "fp16_init_scale": 128,
        "fp16_scale_window": None,
        "fp16_scale_tolerance": 0.0,
        "min_loss_scale": 0.0001,
        "threshold_loss_scale": None,
        "criterion": "cross_entropy",
        "tokenizer": None,
        "bpe": None,
        "optimizer": "nag",
        "lr_scheduler": "fixed",
        "task": "speech_recognition",
        "num_workers": 0,
        "skip_invalid_size_inputs_valid_test": False,
        "max_tokens": 10000000,
        "max_sentences": None,
        "required_batch_size_multiple": 8,
        "dataset_impl": None,
        "gen_subset": "test",
        "num_shards": 1,
        "shard_id": 0,
        "remove_bpe": None,
        "quiet": False,
        "model_overrides": "{}",
        "results_path": None,
        "beam": 40,
        "nbest": 1,
        "max_len_a": 0,
        "max_len_b": 200,
        "min_len": 1,
        "match_source_len": False,
        "no_early_stop": False,
        "unnormalized": False,
        "no_beamable_mm": False,
        "lenpen": 1,
        "unkpen": 0,
        "replace_unk": None,
        "sacrebleu": False,
        "score_reference": False,
        "prefix_size": 0,
        "no_repeat_ngram_size": 0,
        "sampling": False,
        "sampling_topk": -1,
        "sampling_topp": -1.0,
        "temperature": 1.0,
        "diverse_beam_groups": -1,
        "diverse_beam_strength": 0.5,
        "print_alignment": False,
        "ctc": False,
        "rnnt": False,
        "kspmodel": None,
        "wfstlm": None,
        "rnnt_decoding_type": "greedy",
        "lm_weight": 0.2,
        "rnnt_len_penalty": -0.5,
        "momentum": 0.99,
        "weight_decay": 0.0,
        "force_anneal": None,
        "lr_shrink": 0.1,
        "warmup_updates": 0,
    }

    arguments_dict["path"] = os.environ.get("ASR_MODEL_PATH", None)
    arguments_dict["input_file"] = os.environ.get("ASR_INPUT_FILE", None)
    arguments_dict["data"] = os.environ.get("ASR_DATA_PATH", None)
    arguments_dict["user_dir"] = os.environ.get("ASR_USER_DIR", None)
    args = argparse.Namespace(**arguments_dict)

    def test_transcribe_file(self):
        task, generator, models, sp, tgt_dict = setup_asr(self.args, self.logger)
        _, transcription = transcribe_file(
            self.args, task, generator, models, sp, tgt_dict
        )

        expected_transcription = [["THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"]]
        self.assertEqual(transcription, expected_transcription, msg=str(transcription))


if __name__ == "__main__":
    unittest.main()
