import csv
import errno
import gzip
import hashlib
import logging
import os
import sys
import tarfile
import zipfile

import torch
from torch.utils.model_zoo import tqdm

import six
import torchaudio


def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples
    Arguments:
        unicode_csv_data: unicode csv data (see example below)
    Examples:
        >>> from torchtext.utils import unicode_csv_reader
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


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


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


def download_from_url(url, path=None, root=".data", overwrite=False):
    """Download file, with logic (from tensor2tensor) for Google Drive.
    Returns the path to the downloaded file.
    Arguments:
        url: the url of the file
        path: explicitly set the filename, otherwise attempts to
            detect the file name from URL header. (None)
        root: download folder used to store the file in (.data)
        overwrite: overwrite existing files (False)
    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> torchtext.utils.download_from_url(url)
        >>> '.data/validation.tar.gz'
    """

    def _process_response(r, root, filename):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get("Content-length", 0))
        if filename is None:
            d = r.headers["content-disposition"]
            filename = re.findall('filename="(.+)"', d)
            if filename is None:
                raise RuntimeError("Filename could not be autodetected")
            filename = filename[0]
        path = os.path.join(root, filename)
        if os.path.exists(path):
            logging.info("File %s already exists." % path)
            if not overwrite:
                return path
            logging.info("Overwriting file %s." % path)
        logging.info("Downloading file {} to {}.".format(filename, path))
        with open(path, "wb") as file:
            with tqdm(
                total=total_size, unit="B", unit_scale=1, desc=path.split("/")[-1]
            ) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))
        logging.info("File {} downloaded.".format(path))
        return path

    if path is None:
        _, filename = os.path.split(url)
    else:
        root, filename = os.path.split(path)

    if not os.path.exists(root):
        raise RuntimeError(
            "Download directory {} does not exist. " "Did you create it?".format(root)
        )

    if "drive.google.com" not in url:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        return _process_response(response, root, filename)
    else:
        # google drive links get filename from google drive
        filename = None

    logging.info("Downloading from Google Drive; may take a few minutes")
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    return _process_response(response, root, filename)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            else:
                raise e


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests

    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={"id": file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


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
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith((".tar.gz", ".tgz")):
        logging.info("Opening tar file {}.".format(from_path))
        with tarfile.open(from_path, "r") as tar:
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

    elif from_path.endswith(".zip"):
        assert zipfile.is_zipfile(from_path), from_path
        logging.info("Opening zip file {}.".format(from_path))
        with zipfile.ZipFile(from_path, "r") as zfile:
            files = zfile.namelist()
            for file_ in files:
                file_path = os.path.join(to_path, file_)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        return files

    else:
        raise NotImplementedError(
            "We currently only support tar.gz, .tgz and zip achives."
        )


class Cache:
    """
    Wrap a generator so that, whenever a new item is returned, it is saved to disk.
    """

    def __init__(self, generator, location):
        self.generator = generator
        self.location = location

        self._id = id(self)
        self._cache = []
        self._internal_index = 0

    def __iter__(self):
        self._internal_index = 0
        return self

    def __next__(self):
        if self._internal_index < len(self):
            item = self[self._internal_index]
        else:
            item = next(self.generator)

            file = str(self._id) + "-" + str(len(self))
            file = os.path.join(self.location, file)
            self._cache.append(file)

            makedir_exist_ok(self.location)
            torch.save(item, file)

        self._internal_index += 1
        return item

    def __getitem__(self, index):
        file = self._cache[index]
        item = torch.load(file)
        return item

    def __len__(self):
        # Return length of cache
        return len(self._cache)


class Buffer:
    """
    Wrap a generator so as to keep the last few in memory.
    """

    def __init__(self, generator, capacity=10):
        self.generator = generator
        self.capacity = capacity
        self._cache = []
        self._fill()

    def _fill(self):
        # TODO non-blocking after the first item
        while len(self._cache) <= self.capacity:
            self._cache.append(next(self.generator))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._cache.pop(0)
        except IndexError:
            self._fill()
            return self._cache.pop(0)


def download(urls, root_path):
    """
    Download each url to root_path.

    Input: url generator, folder inside archive
    Output: downloaded archive, folder inside archive
    """
    for url, folder in urls:
        torchaudio.datasets.utils.download_url(url, root_path)
        file = os.path.join(root_path, os.path.basename(url))
        yield file, folder


def extract(files):
    """
    Extract each archive to their respective folder.

    Input: (url, folder name inside archive) generator
    Output: path to inside archive
    """
    for file, folder in files:
        torchaudio.datasets.utils.extract_archive(file)
        path = os.path.dirname(file)
        path = os.path.join(path, folder)
        yield path


def walk(paths, extension):
    """
    Walk inside a path recursively to find all files with given extension.

    Input: path
    Output: path, file name identifying a row of data
    """
    for path in paths:
        for _, _, fn in os.walk(path):
            for f in fn:
                if extension in f:
                    yield path, f


def shuffle(generator):
    """
    Shuffle the order of a generator.

    Input: generator
    Output: generator
    """

    # load whole generator in memory
    generator = list(generator)
    # print(len(generator))
    random.shuffle(generator)
    for g in generator:
        yield g


def generator_length(generator):
    """
    Iterate through generator with length.

    Input: generator
    Output: generator
    """

    # load whole generator in memory
    generator = list(generator)
    l = len(generator)
    for g in generator:
        yield g, l


def list_files_recursively(root, suffix, prefix=False, remove_suffix=False):
    """List recursively all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = [f for _, _, fn in os.walk(root) for f in fn if f.endswith(suffix)]

    if remove_suffix:
        files = [f[: -len(suffix)] for f in files]

    if prefix:
        files = [os.path.join(root, d) for d in files]

    return files
