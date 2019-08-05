from __future__ import absolute_import, division, print_function, unicode_literals
import torch.utils.data as data
import os
import os.path
import shutil
import errno
import torch
import torchaudio


class YESNO(data.Dataset):
    """`YesNo Hebrew <http://www.openslr.org/1/>`_ Dataset.

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
    raw_folder = 'yesno/raw'
    processed_folder = 'yesno/processed'
    url = 'http://www.openslr.org/resources/1/waves_yesno.tar.gz'
    dset_path = 'waves_yesno'
    processed_file = 'yesno.pt'

    def __init__(self, root, transform=None, target_transform=None, download=False, dev_mode=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.data = []
        self.labels = []
        self.num_samples = 0
        self.max_len = 0

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self.data, self.labels = torch.load(os.path.join(
            self.root, self.processed_folder, self.processed_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        audio, target = self.data[index], self.labels[index]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.processed_file))

    def download(self):
        """Download the yesno data if it doesn't exist in processed_folder already."""
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
            urllib.request.urlretrieve(url, file_path)
        else:
            print("Tar file already downloaded")
        if not os.path.exists(dset_abs_path):
            with tarfile.open(file_path) as zip_f:
                zip_f.extractall(raw_abs_dir)
        else:
            print("Tar file already extracted")

        if not self.dev_mode:
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        shutil.copyfile(
            os.path.join(dset_abs_path, "README"),
            os.path.join(processed_abs_dir, "YESNO_README")
        )
        audios = [x for x in os.listdir(dset_abs_path) if ".wav" in x]
        print("Found {} audio files".format(len(audios)))
        tensors = []
        labels = []
        lengths = []
        for i, f in enumerate(audios):
            full_path = os.path.join(dset_abs_path, f)
            sig, sr = torchaudio.load(full_path)
            tensors.append(sig)
            lengths.append(sig.size(1))
            labels.append(os.path.basename(f).split(".", 1)[0].split("_"))
        # sort sigs/labels: longest -> shortest
        tensors, labels = zip(*[(b, c) for (a, b, c) in sorted(
            zip(lengths, tensors, labels), key=lambda x: x[0], reverse=True)])
        self.max_len = tensors[0].size(1)
        torch.save(
            (tensors, labels),
            os.path.join(
                self.root,
                self.processed_folder,
                self.processed_file
            )
        )
        if not self.dev_mode:
            shutil.rmtree(raw_abs_dir, ignore_errors=True)

        print('Done!')
