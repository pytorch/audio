import os

from torchaudio.datasets import esc

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


class TestESC(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    data = []
    expected_targets = [0, 0, 38, 38, 40, 40, 41, 41, 10, 10]


    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        print(cls.root_dir)
        base_dir = os.path.join(cls.root_dir, esc.FOLDER_IN_ARCHIVE)
        os.makedirs(base_dir, exist_ok=True)

        # Create a fake metadata file in the meta directory
        meta_dir = os.path.join(base_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        cls.meta_file_path = os.path.join(meta_dir, "esc50.csv")

        with open(cls.meta_file_path, 'w') as fp: 
            fp.write("filename,fold,target,category,esc10,src_file,take\n")

            # Add some file for esc10 and esc50. one for each dataset
            # Note that the esc10 file are in inpair position
            fp.write("0_.wav,1,0,dog,True,100001,A\n")
            fp.write("1_.wav,2,0,dog,False,100002,A\n")
            fp.write("2_.wav,1,38,clock_tick,True,100003,A\n")
            fp.write("3_.wav,2,38,clock_tick,False,100004,A\n")
            fp.write("4_.wav,1,40,helicopter,True,100005,A\n")
            fp.write("5_.wav,2,40,helicopter,False,100006,A\n")
            fp.write("6_.wav,1,41,chainsaw,True,100007,A\n")
            fp.write("7_.wav,2,41,chainsaw,False,100008,A\n")
            fp.write("8_.wav,1,10,rain,True,100009,A\n")
            fp.write("9_.wav,2,10,rain,False,100010,A\n")


        # Create the fake audio file and dir
        audio_dir = os.path.join(base_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        for i in range(10):
            path = os.path.join(audio_dir, "%d_.wav" % i)
            data = get_whitenoise(sample_rate=44100, duration=5, n_channels=1, dtype='int16', seed=i)
            save_wav(path, data, 44100)
            cls.data.append(normalize_wav(data))

    def test_esc10(self):
        dataset = esc.ESC10(self.root_dir, folds=(1, 2))

        # esc10 targets are not linear and the dataset map them from 0 to 9
        target_mapper = {0: 0, 1: 1, 2: 38, 3: 40, 4: 41, 5: 10, 6: 11, 7: 12, 8: 20, 9: 21}
        
        nb_esc10_file = 5
        n_iteration = 0

        for i, (waveform, sample_rate, target) in enumerate(dataset):
            target = target_mapper[target]
            expected_target = self.expected_targets[i*2]  
            expected_data = self.data[i*2]

            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 44100
            assert target == expected_target

            n_iteration += 1

        assert n_iteration == nb_esc10_file

    def test_esc50(self):
        dataset = esc.ESC50(self.root_dir)

        nb_esc50_file = 10
        n_iteration = 0

        for i, (waveform, sample_rate, target) in enumerate(dataset):
            expected_target = self.expected_targets[i]
            expected_data = self.data[i]

            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 44100
            assert target == expected_target

            n_iteration += 1

        assert n_iteration == nb_esc50_file
