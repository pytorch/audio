import pytest
import os


def pytest_collection_modifyitems(config, items):
    fail_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ffmpeg_fail_ids.txt")
    with open(fail_path, 'r') as file:
        fail_ids = set([f.strip() for f in file.readlines()])

    skip_marker = pytest.mark.skip(reason="FFMPEG incompatible with CI runner")

    for item in items:
        if item.nodeid in fail_ids:
            item.add_marker(skip_marker)
