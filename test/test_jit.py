import torch
import torchaudio.functional as F
import torchaudio.transforms as transforms
import unittest

RUN_CUDA = torch.cuda.is_available()
print('Run test with cuda:', RUN_CUDA)


class Test_ScaleJIT(unittest.TestCase):
    def test_torchscript_scale(self):
        @torch.jit.script
        def jit_method(tensor, factor):
            # type: (Tensor, int) -> Tensor
            return F.scale(tensor, factor)

        tensor = torch.rand((10, 1))
        factor = 2

        jit_out = jit_method(tensor, factor)
        py_out = F.scale(tensor, factor)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_scale(self):

        class MyModule(torch.jit.ScriptModule):
            def __init__(self, tensor, factor):
                super(MyModule, self).__init__()
                module = transforms.Scale(factor)
                module.eval()

                self.module = torch.jit.trace(module, (tensor))

            @torch.jit.script_method
            def forward(self, tensor):
                return self.module(tensor)

        tensor = torch.rand((10, 1), device="cuda")
        factor = 2
        model = MyModule(tensor, factor).cuda()

        jit_out = model(tensor)
        py_out = F.scale(tensor, factor)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))


class Test_PadTrimJIT(unittest.TestCase):
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

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_pad_trim(self):

        class MyModule(torch.jit.ScriptModule):
            def __init__(self, tensor, ch_dim, max_len, len_dim, fill_value):
                super(MyModule, self).__init__()
                channels_first = ch_dim == 0
                module = transforms.PadTrim(max_len, fill_value, channels_first=channels_first)
                module.eval()

                self.module = torch.jit.trace(module, (tensor))

            @torch.jit.script_method
            def forward(self, tensor):
                return self.module(tensor)

        tensor = torch.rand((10, 1), device="cuda")
        ch_dim = 1
        max_len = 5
        len_dim = 0
        fill_value = 3.
        model = MyModule(tensor, ch_dim, max_len, len_dim, fill_value).cuda()

        jit_out = model(tensor)
        py_out = F.pad_trim(tensor, ch_dim, max_len, len_dim, fill_value)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))


class Test_DownmixMonoJIT(unittest.TestCase):
    def test_torchscript_downmix_mono(self):
        @torch.jit.script
        def jit_method(tensor, ch_dim):
            # type: (Tensor, int) -> Tensor
            return F.downmix_mono(tensor, ch_dim)

        tensor = torch.rand((10, 1))
        ch_dim = 1

        jit_out = jit_method(tensor, ch_dim)
        py_out = F.downmix_mono(tensor, ch_dim)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_downmix_mono(self):

        class MyModule(torch.jit.ScriptModule):
            def __init__(self, tensor, ch_dim):
                super(MyModule, self).__init__()
                channels_first = ch_dim == 0
                module = transforms.DownmixMono(channels_first)
                module.eval()

                self.module = torch.jit.trace(module, (tensor))

            @torch.jit.script_method
            def forward(self, tensor):
                return self.module(tensor)

        tensor = torch.rand((10, 1), device="cuda")
        ch_dim = 1
        model = MyModule(tensor, ch_dim).cuda()

        jit_out = model(tensor)
        py_out = F.downmix_mono(tensor, ch_dim)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))


class Test_LC2CLJIT(unittest.TestCase):
    def test_torchscript_LC2CL(self):
        @torch.jit.script
        def jit_method(tensor):
            # type: (Tensor) -> Tensor
            return F.LC2CL(tensor)

        tensor = torch.rand((10, 1))

        jit_out = jit_method(tensor)
        py_out = F.LC2CL(tensor)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_LC2CL(self):

        class MyModule(torch.jit.ScriptModule):
            def __init__(self, tensor):
                super(MyModule, self).__init__()
                module = transforms.LC2CL(tensor)
                module.eval()

                self.module = torch.jit.trace(module, (tensor))

            @torch.jit.script_method
            def forward(self, tensor):
                return self.module(tensor)

        tensor = torch.rand((10, 1), device="cuda")
        model = MyModule(tensor).cuda()

        jit_out = model(tensor)
        py_out = F.LC2CL(tensor)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))

if __name__ == '__main__':
    unittest.main()
