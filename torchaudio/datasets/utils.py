import errno
import gzip
import hashlib
import os
import os.path
import tarfile
import zipfile

import torch
from torch.utils.model_zoo import tqdm


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


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path, os.path.splitext(os.path.basename(from_path))[0]
        )
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


class Cache:
    """
    Wrap a generator so that, whenever a new item is returned, it is saved to disk in a pickle.
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
            with open(file, "wb") as file:
                pickle.dump(item, file)

        self._internal_index += 1
        return item

    def __getitem__(self, index):
        file = self._cache[index]
        with open(file, "rb") as file:
            item = pickle.load(file)
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
        while len(self._cache) <= self.capacity:
            self._cache.append(next(self.generator))

    def __getitem__(self, n):
        self._fill()
        return self._cache[n]

    def __iter__(self):
        return self

    def __next__(self):
        item = self._cache.pop(0)
        self._fill()
        return item


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
        for dp, dn, fn in os.walk(path):
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
