from __future__ import absolute_import, division, print_function, unicode_literals
import os
from shutil import copytree
import backports.tempfile as tempfile


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
