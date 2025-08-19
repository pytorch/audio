import argparse
import hashlib
import json
import logging
import os
import platform
import stat
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

# String representing the host platform (e.g. Linux, Darwin).
HOST_PLATFORM = platform.system()

# PyTorch directory root
try:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        check=True,
    )
    PYTORCH_ROOT = result.stdout.decode("utf-8").strip()
except subprocess.CalledProcessError:
    # If git is not installed, compute repo root as 3 folders up from this file
    path_ = os.path.abspath(__file__)
    for _ in range(4):
        path_ = os.path.dirname(path_)
    PYTORCH_ROOT = path_

DRY_RUN = False


def compute_file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file and return it as a hex string."""
    # If the file doesn't exist, return an empty string.
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha256()

    # Open the file in binary mode and hash it.
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)

    # Return the hash as a hexadecimal string.
    return hash.hexdigest()


def report_download_progress(
    chunk_number: int, chunk_size: int, file_size: int
) -> None:
    """
    Pretty printer for file download progress.
    """
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


def check(binary_path: Path, reference_hash: str) -> bool:
    """Check whether the binary exists and is the right one.

    If there is hash difference, delete the actual binary.
    """
    if not binary_path.exists():
        logging.info(f"{binary_path} does not exist.")
        return False

    existing_binary_hash = compute_file_sha256(str(binary_path))
    if existing_binary_hash == reference_hash:
        return True

    logging.warning(
        textwrap.dedent(
            f"""\
            Found binary hash does not match reference!

            Found hash: {existing_binary_hash}
            Reference hash: {reference_hash}

            Deleting {binary_path} just to be safe.
            """
        )
    )
    if DRY_RUN:
        logging.critical(
            "In dry run mode, so not actually deleting the binary. But consider deleting it ASAP!"
        )
        return False

    try:
        binary_path.unlink()
    except OSError as e:
        logging.critical(f"Failed to delete binary: {e}")
        logging.critical(
            "Delete this binary as soon as possible and do not execute it!"
        )

    return False


def download(
    name: str,
    output_dir: str,
    url: str,
    reference_bin_hash: str,
) -> bool:
    """
    Download a platform-appropriate binary if one doesn't already exist at the expected location and verifies
    that it is the right binary by checking its SHA256 hash against the expected hash.
    """
    # First check if we need to do anything
    binary_path = Path(output_dir, name)
    if check(binary_path, reference_bin_hash):
        logging.info(f"Correct binary already exists at {binary_path}. Exiting.")
        return True

    # Create the output folder
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the binary
    logging.info(f"Downloading {url} to {binary_path}")

    if DRY_RUN:
        logging.info("Exiting as there is nothing left to do in dry run mode")
        return True

    urllib.request.urlretrieve(
        url,
        binary_path,
        reporthook=report_download_progress if sys.stdout.isatty() else None,
    )

    logging.info(f"Downloaded {name} successfully.")

    # Check the downloaded binary
    if not check(binary_path, reference_bin_hash):
        logging.critical(f"Downloaded binary {name} failed its hash check")
        return False

    # Ensure that exeuctable bits are set
    mode = os.stat(binary_path).st_mode
    mode |= stat.S_IXUSR
    os.chmod(binary_path, mode)

    logging.info(f"Using {name} located at {binary_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="downloads and checks binaries from s3",
    )
    parser.add_argument(
        "--config-json",
        required=True,
        help="Path to config json that describes where to find binaries and hashes",
    )
    parser.add_argument(
        "--linter",
        required=True,
        help="Which linter to initialize from the config json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="place to put the binary",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="name of binary",
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        help="do not download, just print what would be done",
    )

    args = parser.parse_args()
    if args.dry_run == "0":
        DRY_RUN = False
    else:
        DRY_RUN = True

    logging.basicConfig(
        format="[DRY_RUN] %(levelname)s: %(message)s"
        if DRY_RUN
        else "%(levelname)s: %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    config = json.load(open(args.config_json))
    config = config[args.linter]

    # If the host platform is not in platform_to_hash, it is unsupported.
    if HOST_PLATFORM not in config:
        logging.error(f"Unsupported platform: {HOST_PLATFORM}")
        exit(1)

    url = config[HOST_PLATFORM]["download_url"]
    hash = config[HOST_PLATFORM]["hash"]

    ok = download(args.output_name, args.output_dir, url, hash)
    if not ok:
        logging.critical(f"Unable to initialize {args.linter}")
        sys.exit(1)
