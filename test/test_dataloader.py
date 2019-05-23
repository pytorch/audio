# import unittest
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchaudio
# import math
# import os
#
#
# class TORCHAUDIODS(Dataset):
#
#     test_dirpath = os.path.dirname(os.path.realpath(__file__))
#
#     def __init__(self):
#         self.asset_dirpath = os.path.join(self.test_dirpath, "assets")
#         sound_files = list(filter(lambda x: '.wav' in x or '.mp3' in x, os.listdir(self.asset_dirpath)))
#         self.data = [os.path.join(self.asset_dirpath, fn) for fn in sound_files]
#         self.si, self.ei = torchaudio.info(os.path.join(self.asset_dirpath, "sinewave.wav"))
#         self.si.precision = 16
#         self.E = torchaudio.sox_effects.SoxEffectsChain()
#         self.E.append_effect_to_chain("rate", [self.si.rate])  # resample to 16000hz
#         self.E.append_effect_to_chain("channels", [self.si.channels])  # mono signal
#         self.E.append_effect_to_chain("trim", [0, "16000s"])  # first 16000 samples of audio
#
#     def __getitem__(self, index):
#
#         fn = self.data[index]
#         self.E.set_input_file(fn)
#         x, sr = self.E.sox_build_flow_effects()
#         return x
#
#     def __len__(self):
#         return len(self.data)
#
#
# class Test_DataLoader(unittest.TestCase):
#     def test_1(self):
#         expected_size = (2, 1, 16000)
#         ds = TORCHAUDIODS()
#         dl = DataLoader(ds, batch_size=2)
#         for x in dl:
#             self.assertTrue(x.size() == expected_size)
#
# if __name__ == '__main__':
#     torchaudio.initialize_sox()
#     unittest.main()
#     torchaudio.shutdown_sox()
