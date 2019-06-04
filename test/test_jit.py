import torch
import torchaudio.functional as F
import torchaudio.transforms as transforms
import unittest

RUN_CUDA = torch.cuda.is_available()
print('Run test with cuda:', RUN_CUDA)


class Test_JIT(unittest.TestCase):
    def _get_script_module(self, f, *args):
        # takes a transform function `f` and wraps it in a script module
        class MyModule(torch.jit.ScriptModule):
            def __init__(self):
                super(MyModule, self).__init__()
                self.module = f(*args)
                self.module.eval()

            @torch.jit.script_method
            def forward(self, tensor):
                return self.module(tensor)

        return MyModule()

    def _test_script_module(self, tensor, f, *args):
        # tests a script module that wraps a transform function `f` by feeding
        # the tensor into the forward function
        jit_out = self._get_script_module(f, *args).cuda()(tensor)
        py_out = f(*args).cuda()(tensor)

        self.assertTrue(torch.allclose(jit_out, py_out))

    def test_torchscript_scale(self):
        @torch.jit.script
        def jit_method(tensor, factor):
            # type: (Tensor, int) -> Tensor
            return F.scale(tensor, factor)

        tensor = torch.rand((10, 1))
        factor = 2

        jit_out = jit_method(tensor, factor)
        py_out = F.scale(tensor, factor)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_scale(self):
        tensor = torch.rand((10, 1), device="cuda")

        self._test_script_module(tensor, transforms.Scale)

    def test_torchscript_pad_trim(self):
        @torch.jit.script
        def jit_method(tensor, ch_dim, max_len, len_dim, fill_value):
            # type: (Tensor, int, int, int, float) -> Tensor
            return F.pad_trim(tensor, ch_dim, max_len, len_dim, fill_value)

        tensor = torch.rand((10, 1))
        ch_dim = 1
        max_len = 5
        len_dim = 0
        fill_value = 3.

        jit_out = jit_method(tensor, ch_dim, max_len, len_dim, fill_value)
        py_out = F.pad_trim(tensor, ch_dim, max_len, len_dim, fill_value)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_pad_trim(self):
        tensor = torch.rand((1, 10), device="cuda")
        max_len = 5

        self._test_script_module(tensor, transforms.PadTrim, max_len)

    def test_torchscript_downmix_mono(self):
        @torch.jit.script
        def jit_method(tensor, ch_dim):
            # type: (Tensor, int) -> Tensor
            return F.downmix_mono(tensor, ch_dim)

        tensor = torch.rand((10, 1))
        ch_dim = 1

        jit_out = jit_method(tensor, ch_dim)
        py_out = F.downmix_mono(tensor, ch_dim)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_downmix_mono(self):
        tensor = torch.rand((1, 10), device="cuda")

        self._test_script_module(tensor, transforms.DownmixMono)

    def test_torchscript_LC2CL(self):
        @torch.jit.script
        def jit_method(tensor):
            # type: (Tensor) -> Tensor
            return F.LC2CL(tensor)

        tensor = torch.rand((10, 1))

        jit_out = jit_method(tensor)
        py_out = F.LC2CL(tensor)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_LC2CL(self):
        tensor = torch.rand((10, 1), device="cuda")

        self._test_script_module(tensor, transforms.LC2CL)

    def test_torchscript_spectrogram(self):
        @torch.jit.script
        def jit_method(sig, pad, window, n_fft, hop, ws, power, normalize):
            # type: (Tensor, int, Tensor, int, int, int, int, bool) -> Tensor
            return F.spectrogram(sig, pad, window, n_fft, hop, ws, power, normalize)

        tensor = torch.rand((1, 1000))
        n_fft = 400
        ws = 400
        hop = 200
        pad = 0
        window = torch.hann_window(ws)
        power = 2
        normalize = False

        jit_out = jit_method(tensor, pad, window, n_fft, hop, ws, power, normalize)
        py_out = F.spectrogram(tensor, pad, window, n_fft, hop, ws, power, normalize)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_Spectrogram(self):
        tensor = torch.rand((1, 1000), device="cuda")

        self._test_script_module(tensor, transforms.Spectrogram)

    def test_torchscript_create_fb_matrix(self):
        @torch.jit.script
        def jit_method(n_stft, sr, f_min, f_max, n_mels):
            # type: (int, int, float, Optional[float], int) -> Tensor
            return F.create_fb_matrix(n_stft, sr, f_min, f_max, n_mels)

        n_stft = 100
        sr = 16000
        f_min = 0.
        f_max = 20.
        n_mels = 10

        jit_out = jit_method(n_stft, sr, f_min, f_max, n_mels)
        py_out = F.create_fb_matrix(n_stft, sr, f_min, f_max, n_mels)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_MelScale(self):
        spec_f = torch.rand((1, 6, 201), device="cuda")

        self._test_script_module(spec_f, transforms.MelScale)

    def test_torchscript_spectrogram_to_DB(self):
        @torch.jit.script
        def jit_method(spec, multiplier, amin, db_multiplier, top_db):
            # type: (Tensor, float, float, float, Optional[float]) -> Tensor
            return F.spectrogram_to_DB(spec, multiplier, amin, db_multiplier, top_db)

        spec = torch.rand((10, 1))
        multiplier = 10.
        amin = 1e-10
        db_multiplier = 0.
        top_db = 80.

        jit_out = jit_method(spec, multiplier, amin, db_multiplier, top_db)
        py_out = F.spectrogram_to_DB(spec, multiplier, amin, db_multiplier, top_db)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_SpectrogramToDB(self):
        spec = torch.rand((10, 1), device="cuda")

        self._test_script_module(spec, transforms.SpectrogramToDB)

    def test_torchscript_create_dct(self):
        @torch.jit.script
        def jit_method(n_mfcc, n_mels, norm):
            # type: (int, int, Optional[str]) -> Tensor
            return F.create_dct(n_mfcc, n_mels, norm)

        n_mfcc = 40
        n_mels = 128
        norm = 'ortho'

        jit_out = jit_method(n_mfcc, n_mels, norm)
        py_out = F.create_dct(n_mfcc, n_mels, norm)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_MFCC(self):
        tensor = torch.rand((1, 1000), device="cuda")

        self._test_script_module(tensor, transforms.MFCC)

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_MelSpectrogram(self):
        tensor = torch.rand((1, 1000), device="cuda")

        self._test_script_module(tensor, transforms.MelSpectrogram)

    def test_torchscript_BLC2CBL(self):
        @torch.jit.script
        def jit_method(tensor):
            # type: (Tensor) -> Tensor
            return F.BLC2CBL(tensor)

        tensor = torch.rand((10, 1000, 1))

        jit_out = jit_method(tensor)
        py_out = F.BLC2CBL(tensor)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_BLC2CBL(self):
        tensor = torch.rand((10, 1000, 1), device="cuda")

        self._test_script_module(tensor, transforms.BLC2CBL)

    def test_torchscript_mu_law_encoding(self):
        @torch.jit.script
        def jit_method(tensor, qc):
            # type: (Tensor, int) -> Tensor
            return F.mu_law_encoding(tensor, qc)

        tensor = torch.rand((10, 1))
        qc = 256

        jit_out = jit_method(tensor, qc)
        py_out = F.mu_law_encoding(tensor, qc)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_MuLawEncoding(self):
        tensor = torch.rand((10, 1), device="cuda")

        self._test_script_module(tensor, transforms.MuLawEncoding)

    def test_torchscript_mu_law_expanding(self):
        @torch.jit.script
        def jit_method(tensor, qc):
            # type: (Tensor, int) -> Tensor
            return F.mu_law_expanding(tensor, qc)

        tensor = torch.rand((10, 1))
        qc = 256

        jit_out = jit_method(tensor, qc)
        py_out = F.mu_law_expanding(tensor, qc)

        self.assertTrue(torch.allclose(jit_out, py_out))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_MuLawExpanding(self):
        tensor = torch.rand((10, 1), device="cuda")

        self._test_script_module(tensor, transforms.MuLawExpanding)


if __name__ == '__main__':
    unittest.main()
