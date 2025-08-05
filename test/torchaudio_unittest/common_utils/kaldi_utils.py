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
    If the result has not yet been computed, run the provided Kaldi command,
    pass a tensor and get the resulting tensor

    Args:
        command (list of str): The command with arguments
        input_type (str): 'ark' or 'scp'
        input_value (Tensor for 'ark', string for 'scp'): The input to pass.
            Must be a path to an audio file for 'scp'.
    """
    path = Path(f"torchaudio_unittest/assets/kaldi_expected_results/{request}.pt")
    if os.path.exists(path):
        return torch.load(path)
    import kaldi_io

    key = "foo"
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if input_type == "ark":
        kaldi_io.write_mat(process.stdin, input_value.cpu().numpy(), key=key)
    elif input_type == "scp":
        process.stdin.write(f"{key} {input_value}".encode("utf8"))
    else:
        raise NotImplementedError("Unexpected type")
    process.stdin.close()
    result = dict(kaldi_io.read_mat_ark(process.stdout))["foo"]
    torch_result = torch.from_numpy(result.copy())  # copy supresses some torch warning
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_result, path)
    return torch_result
