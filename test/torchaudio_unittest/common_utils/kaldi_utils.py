import subprocess

import torch
import os
from pathlib import Path


def convert_args(**kwargs):
    args = []
    for key, value in kwargs.items():
        if key == "sample_rate":
            key = "sample_frequency"
        key = "--" + key.replace("_", "-")
        value = str(value).lower() if value in [True, False] else str(value)
        args.append("%s=%s" % (key, value))
    return args


def run_kaldi(request, command, input_type, input_value):
    """Get the precomputed result of running a Kaldi command.
    In the commented out code, if the result has not yet been computed,
    run the provided Kaldi command (passing a tensor and getting the result).
    This is used to check that torchaudio functionality matches corresponding
    Kaldi functionality.

    Args:
        command (list of str): The command with arguments
        input_type (str): 'ark' or 'scp'
        input_value (Tensor for 'ark', string for 'scp'): The input to pass.
            Must be a path to an audio file for 'scp'.
    """
    test_dir = Path(__file__).parent.parent.resolve()
    expected_results_folder = test_dir / "assets" / "kaldi_expected_results"
    mocked_results = f"{expected_results_folder / request}.pt"
    if os.path.exists(mocked_results):
        return torch.load(mocked_results)

    # import kaldi_io
    # key = "foo"
    # process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # if input_type == "ark":
    #     kaldi_io.write_mat(process.stdin, input_value.cpu().numpy(), key=key)
    # elif input_type == "scp":
    #     process.stdin.write(f"{key} {input_value}".encode("utf8"))
    # else:
    #     raise NotImplementedError("Unexpected type")
    # process.stdin.close()
    # result = dict(kaldi_io.read_mat_ark(process.stdout))["foo"]
    # torch_result = torch.from_numpy(result.copy())  # copy supresses some torch warning
    # mocked_results.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(torch_result, mocked_results)
    # return torch_result
