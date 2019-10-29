import csv
import errno
import gzip
import hashlib
import logging
import os
import sys
import tarfile
import threading
import zipfile
from queue import Queue

import six
import torch
import torchaudio
from six.moves import urllib
from torch.utils.data import Dataset
from torch.utils.model_zoo import tqdm


def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples
    Arguments:
        unicode_csv_data: unicode csv data (see example below)
    Examples:
        >>> from torchaudio.datasets.utils import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)
    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    if six.PY2:
        # csv.py doesn't do Unicode; encode temporarily as UTF-8:
        csv_reader = csv.reader(utf_8_encoder(unicode_csv_data), **kwargs)
        for row in csv_reader:
            # decode UTF-8 back to Unicode, cell by cell:
            yield [cell.decode("utf-8") for cell in row]
    else:
        for line in csv.reader(unicode_csv_data, **kwargs):
            yield line


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, download_folder, hash_value=None, hash_type="sha256"):
    """Execute the correct download operation.
    Depending on the size of the file online and offline, resume the
    download if the file offline is smaller than online.

    Args:
        url (str): Url.
        download_folder (str): Folder to download file.
        hash_value (str): Hash for url.
        hash_type (str): Hash type.
    """

    filepath = os.path.join(download_folder, os.path.basename(url))

    req = urllib.request.Request(url)
    if os.path.exists(filepath):
        mode = "ab"
        local_size = os.path.getsize(filepath)

        # If the file exists, then download only the remainder
        req.headers["Range"] = "bytes={}-".format(local_size)
    else:
        mode = "wb"
        local_size = 0

    # If we already have the whole file, there is no need to download it again
    url_size = int(urllib.request.urlopen(url).info().get("Content-Length", -1))
    if url_size == local_size:
        if hash_value and not validate_download_url(filepath, hash_value, hash_type):
            raise RuntimeError(
                "The hash of {} does not match. Delete the file manually and retry.".format(
                    filepath
                )
            )

        return

    with open(filepath, mode) as fpointer, urllib.request.urlopen(
        req
    ) as upointer, tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, total=url_size
    ) as pbar:

        num_bytes = 0
        block_size = 32 * 1024
        while True:
            chunk = upointer.read(block_size)
            if not chunk:
                break
            fpointer.write(chunk)
            num_bytes += len(chunk)
            pbar.update(len(chunk))


def validate_download_url(filepath, hash_value, hash_type="sha256"):
    """Validate a given file with its hash.

    Args:
        url (str): Url.
        download_folder (str): Folder to download file.
        hash_value (str): Hash for url.
        hash_type (str): Hash type.
    """

    if hash_type == "sha256":
        sha = hashlib.sha256()
    elif hash_type == "md5":
        sha = hashlib.md5()
    else:
        raise ValueError

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1000 * 1000)  # 1MB so that memory is not exhausted
            if not chunk:
                break
            sha.update(chunk)

    return sha.hexdigest() == hash_value


def extract_archive(from_path, to_path=None, overwrite=False):
    """Extract archive.
    Arguments:
        from_path: the path of the archive.
        to_path: the root path of the extraced files (directory of from_path)
        overwrite: overwrite existing files (False)
    Returns:
        List of paths to extracted files even if not overwritten.
    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchaudio.datasets.utils.download_from_url(url, from_path)
        >>> torchaudio.datasets.utils.extract_archive(from_path, to_path)
    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    try:
        with tarfile.open(from_path, "r") as tar:
            logging.info("Opened tar file {}.".format(from_path))
            files = []
            for file_ in tar:
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            return files
    except tarfile.ReadError:
        pass

    try:
        with zipfile.ZipFile(from_path, "r") as zfile:
            logging.info("Opened zip file {}.".format(from_path))
            files = zfile.namelist()
            for file_ in files:
                file_path = os.path.join(to_path, file_)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        return files
    except zipfile.BadZipFile:
        pass

    raise NotImplementedError("We currently only support tar.gz, tgz, and zip achives.")


def walk_files(root, suffix, prefix=False, remove_suffix=False):
    """List recursively all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """

    root = os.path.expanduser(root)

    for _, _, fn in os.walk(root):
        for f in fn:
            if f.endswith(suffix):

                if remove_suffix:
                    f = f[: -len(suffix)]

                if prefix:
                    f = os.path.join(root, f)

                yield f


class _DiskCache(Dataset):
    """
    Wrap a dataset so that, whenever a new item is returned, it is saved to disk.
    """

    def __init__(self, dataset, location=".cached"):
        self.dataset = dataset
        self.location = location

        self._id = id(self)
        self._cache = [None] * len(dataset)

    def __getitem__(self, n):
        if self._cache[n]:
            f = self._cache[n]
            return torch.load(f)

        f = str(self._id) + "-" + str(n)
        f = os.path.join(self.location, f)
        item = self.dataset[n]

        self._cache[n] = f
        makedir_exist_ok(self.location)
        torch.save(item, f)

        return item

    def __len__(self):
        return len(self.dataset)


def diskcache_iterator(dataset, location=".cached"):
    return _DiskCache(dataset, location)


class _ThreadedIterator(threading.Thread):
    """
    Prefetch the next queue_length items from iterator in a background thread.

    Example:
    >> for i in bg_iterator(range(10)):
    >>     print(i)
    """

    class _End:
        pass

    def __init__(self, generator, maxsize):
        threading.Thread.__init__(self)
        self.queue = Queue(maxsize)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(self._End)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item == self._End:
            raise StopIteration
        return next_item


def bg_iterator(iterable, maxsize):
    return _ThreadedIterator(iterable, maxsize=maxsize)
