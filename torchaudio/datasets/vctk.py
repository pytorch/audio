from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import shutil
import errno
import torch
import torchaudio

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_manifest(dir):
    audios = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = path
                    audios.append(item)
    return audios


def read_audio(fp, downsample=True):
    sig, sr = torchaudio.load(fp)
    if downsample:
        # 48khz -> 16 khz
        if sig.size(0) % 3 == 0:
            sig = sig[::3].contiguous()
        else:
            sig = sig[:-(sig.size(0) % 3):3].contiguous()
    return sig, sr


def load_txts(dir):
    """Create a dictionary with all the text of the audio transcriptions."""
    utterences = dict()
    txts = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if fname.endswith(".txt"):
                    with open(os.path.join(root, fname), "r") as f:
                        fname_no_ext = os.path.basename(
                            fname).rsplit(".", 1)[0]
                        utterences[fname_no_ext] = f.readline()
    return utterences


class VCTK(data.Dataset):
    """`VCTK <http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_ Dataset.
    `alternate url <http://datashare.is.ed.ac.uk/handle/10283/2651>`

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Scale``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        dev_mode(bool, optional): if true, clean up is not performed on downloaded
            files.  Useful to keep raw audio and transcriptions.
    """
    raw_folder = 'vctk/raw'
    processed_folder = 'vctk/processed'
    url = 'http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz'
    dset_path = 'VCTK-Corpus'

    def __init__(self, root, downsample=True, transform=None, target_transform=None, download=False, dev_mode=False):
        self.root = os.path.expanduser(root)
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.data = []
        self.labels = []
        self.chunk_size = 1000
        self.num_samples = 0
        self.max_len = 0
        self.cached_pt = 0

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self._read_info()
        self.data, self.labels = torch.load(os.path.join(
            self.root, self.processed_folder, "vctk_{:04d}.pt".format(self.cached_pt)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = int(index // self.chunk_size)
            self.data, self.labels = torch.load(os.path.join(
                self.root, self.processed_folder, "vctk_{:04d}.pt".format(self.cached_pt)))
        index = index % self.chunk_size
        audio, target = self.data[index], self.labels[index]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self):
        return self.num_samples

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "vctk_info.txt"))

    def _write_info(self, num_items):
        info_path = os.path.join(
            self.root, self.processed_folder, "vctk_info.txt")
        with open(info_path, "w") as f:
            f.write("num_samples,{}\n".format(num_items))
            f.write("max_len,{}\n".format(self.max_len))

    def _read_info(self):
        info_path = os.path.join(
            self.root, self.processed_folder, "vctk_info.txt")
        with open(info_path, "r") as f:
            self.num_samples = int(f.readline().split(",")[1])
            self.max_len = int(f.readline().split(",")[1])

    def download(self):
        """Download the VCTK data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        raw_abs_dir = os.path.join(self.root, self.raw_folder)
        processed_abs_dir = os.path.join(self.root, self.processed_folder)
        dset_abs_path = os.path.join(
            self.root, self.raw_folder, self.dset_path)

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = self.url
        print('Downloading ' + url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.isfile(file_path):
            data = urllib.request.urlopen(url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
        if not os.path.exists(dset_abs_path):
            with tarfile.open(file_path) as zip_f:
                zip_f.extractall(raw_abs_dir)
        else:
            print("Using existing raw folder")
        if not self.dev_mode:
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        shutil.copyfile(
            os.path.join(dset_abs_path, "COPYING"),
            os.path.join(processed_abs_dir, "VCTK_COPYING")
        )
        audios = make_manifest(dset_abs_path)
        utterences = load_txts(dset_abs_path)
        self.max_len = 0
        print("Found {} audio files and {} utterences".format(
            len(audios), len(utterences)))
        for n in range(len(audios) // self.chunk_size + 1):
            tensors = []
            labels = []
            lengths = []
            st_idx = n * self.chunk_size
            end_idx = st_idx + self.chunk_size
            for i, f in enumerate(audios[st_idx:end_idx]):
                txt_dir = os.path.dirname(f).replace("wav48", "txt")
                if os.path.exists(txt_dir):
                    f_rel_no_ext = os.path.basename(f).rsplit(".", 1)[0]
                    sig = read_audio(f, downsample=self.downsample)[0]
                    tensors.append(sig)
                    lengths.append(sig.size(0))
                    labels.append(utterences[f_rel_no_ext])
                    self.max_len = sig.size(0) if sig.size(
                        0) > self.max_len else self.max_len
            # sort sigs/labels: longest -> shortest
            tensors, labels = zip(*[(b, c) for (a, b, c) in sorted(
                zip(lengths, tensors, labels), key=lambda x: x[0], reverse=True)])
            data = (tensors, labels)
            torch.save(
                data,
                os.path.join(
                    self.root,
                    self.processed_folder,
                    "vctk_{:04d}.pt".format(n)
                )
            )
        self._write_info((n * self.chunk_size) + i + 1)
        if not self.dev_mode:
            shutil.rmtree(raw_abs_dir, ignore_errors=True)

        print('Done!')
