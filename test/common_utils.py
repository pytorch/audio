import os
from shutil import copytree
import tempfile
import torch


TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def create_temp_assets_dir():
    """
    Creates a temporary directory and moves all files from test/assets there.
    Returns a Tuple[string, TemporaryDirectory] which is the folder path
    and object.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    copytree(os.path.join(TEST_DIR_PATH, "assets"),
             os.path.join(tmp_dir.name, "assets"))
    return tmp_dir.name, tmp_dir


class RandomTensorGenerator:
    """ Generates random tensors given a seed and size
    https://en.wikipedia.org/wiki/Linear_congruential_generator
    X_{n + 1} = (a * X_n + c) % m
    Using Borland C/C++ values
    """
    a = 22695477
    c = 1
    m = 2 ** 32

    def __init__(self, seed, size):
        # seed is an int and size is a tuple of ints
        torch.manual_seed(seed)
        self.x_n = torch.randint(low=0, high=self.m, size=size)

    def rand_int_tensor(self, rand_min=0, rand_max=2 ** 32):
        # returns an integer tensor between [rand_min, rand_max)
        return torch.floor(self.rand_float_tensor() * (rand_max - rand_min)) + rand_min

    def rand_float_tensor(self):
        # returns a tensor between [0, 1)
        self.x_n = (self.a * self.x_n + self.c) % self.m
        return self.x_n.float() / self.m
