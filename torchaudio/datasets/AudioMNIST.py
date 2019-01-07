from __future__ import print_function
from glob import glob
import torch.utils.data as data
import os
import os.path
import errno
import torch
import torchaudio
import json
import random
import math
import urllib
import zipfile
from tqdm import tqdm


class AudioMNIST(data.Dataset):
    """ `AudioMNIST <https://github.com/soerenab/AudioMNIST/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and ``processed/test.pt`` exist.
        split (string): The dataset has 3 different splits: ``digit``,
            ``speaker``, and ``gender``. This argument specifies which one to use.
        train (bool, optonal): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an audio
            tensor and returns a transformed version. E.g. ``transforms.Scale``
        target_transform (callable, optional) A function/transform that takes in the
            target and transforms it.
        dev_mode(bool, optional): if true, clean up is not performed on downloaded
            files. Useful to keep raw audio.
    """
    url = 'https://github.com/soerenab/AudioMNIST/archive/master.zip'
    splits = ('digit', 'speaker', 'gender')
    raw_folder = 'raw'
    processed_folder = 'processed'

    def __init__(self, root, split, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        # if not self._check_exists():
        #     raise RuntimeError('Dataset not found. You can use download=True to download it.')

        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))

        self.split = split
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        :param index (int): Index
        :return: tuple: (audio, target) where target is index of the target class.
        """
        if self.train:
            audio, target = self.train_data[index], self.train_labels[index]
        else:
            audio, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
                os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    @staticmethod
    def _training_file(split):
        return 'training_{}.pt'.format(split)

    @staticmethod
    def _test_file(split):
        return 'test_{}.pt'.format(split)

    def download(self):
        """ Download the AudioMNIST data if it doesn't exist in processed_folder already."""

        # download files
        try:
            os.makedirs(os.path.join(self.root, os.path.join(self.raw_folder)))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = self.url
        print('Downloading from ' + url)
        filename = url.rpartition('/')[2]
        raw_folder = os.path.join(self.root, self.raw_folder)
        file_path = os.path.join(raw_folder, filename)
        if not os.path.exists(os.path.join(raw_folder, 'AudioMNIST-master')):
            d = urllib.request.urlopen(url)
            with open(file_path, 'wb') as f:
                f.write(d.read())
            print('Extracting zip archive')
            with zipfile.ZipFile(file_path) as zip_f:
                zip_f.extractall(raw_folder)
            os.unlink(file_path)
            print('Done')
        else:
            print("File already exists.")
            pass

        try:
            os.makedirs(os.path.join(self.root, os.path.join(self.processed_folder)))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # Process and save as torch files
        for split in self.splits:
            if split != 'gender':
                print('{} split is not ready yet.'.format(split))
            else:
                print('Processing ' + split)

                # We don't have good predefined splits, so for now randomly select.
                # For gender task we don't want the same speaker in test and train, so shuffle on the IDs.
                metadata = generate_metadata_list(raw_folder)
                id_list = list({item['speaker-id']: item for item in metadata})
                random.shuffle(id_list)
                train_list = id_list[:math.floor(len(id_list)*0.8)]
                train_audio_paths = []
                train_labels = []

                test_audio_paths = []
                test_labels = []

                for item in tqdm(metadata):
                    if item['speaker-id'] in train_list:
                        train_audio_paths.append(item['path'])
                        train_labels.append(item['gender'])
                    else:
                        test_audio_paths.append(item['path'])
                        test_labels.append(item['gender'])

                training_set = (
                    read_audio_file(train_audio_paths),
                    train_labels
                )
                test_set = (
                    read_audio_file(test_audio_paths),
                    test_labels
                )

                with open(os.path.join(self.root, self.processed_folder, self._training_file(split)), 'wb') as f:
                    torch.save(training_set, f)
                with open(os.path.join(self.root, self.processed_folder, self._test_file(split)), 'wb') as f:
                    torch.save(test_set, f)


def read_audio_file(paths):
    audio_data = []
    for path in paths:
        audio_data.append(torchaudio.load(path))
    return audio_data


def generate_metadata_list(raw_folder):
    """ Returns a list of all the data """
    data_path = os.path.join(raw_folder, 'AudioMNIST-master/data/')
    meta_file = os.path.join(data_path, 'audioMNIST_meta.txt')
    with open(meta_file) as f:
        meta_data = json.load(f)

    meta_list = []
    for sid in meta_data.keys():

        # gender
        gender = meta_data[sid]['gender']

        # paths
        glob_pattern = os.path.join(data_path, sid, '*')
        paths = glob(glob_pattern)

        # digits
        digits = [d[0] for d in os.listdir(os.path.join(data_path, sid))]

        # create dict id-gender-path-digit
        for i in range(len(paths)):
            entry = {
                'path': paths[i],
                'speaker-id': sid,
                'gender': gender,
                'digit': digits[i]
            }
            # print(entry)
            meta_list.append(entry)

    return meta_list


# https://github.com/soerenab/AudioMNIST/archive/master.zip