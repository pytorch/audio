import torch
import torchaudio.functional as F
import torchaudio.transforms as transforms
import unittest

RUN_CUDA = torch.cuda.is_available()
print('Run test with cuda:', RUN_CUDA)


class Test_ScaleJIT(unittest.TestCase):
    def test_torchscript_scale(self):
        @torch.jit.script
        def jit_scale(tensor, factor):
            # type: (Tensor, int) -> Tensor
            return F.scale(tensor, factor)

        n = 10
        tensor = torch.rand((n, 1))
        factor = 2

        jit_out = jit_scale(tensor, factor)
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

        n = 10
        tensor = torch.rand((n, 1), device="cuda")
        factor = 2
        model = MyModule(tensor, factor).cuda()

        jit_out = model(tensor)
        py_out = F.scale(tensor, factor)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))


class Test_PadTrimJIT(unittest.TestCase):
    def test_torchscript_pad_trim(self):
        @torch.jit.script
        def jit_pad_trim(tensor, factor):
            # type: (Tensor, int) -> Tensor
            return F.scale(tensor, factor)

        n = 10
        tensor = torch.rand((n, 1))
        factor = 2

        jit_out = jit_scale(tensor, factor)
        py_out = F.scale(tensor, factor)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_scriptmodule_pad_trim(self):

        class MyModule(torch.jit.ScriptModule):
            def __init__(self, tensor, factor):
                super(MyModule, self).__init__()
                module = transforms.Scale(factor)
                module.eval()

                self.module = torch.jit.trace(module, (tensor))

            @torch.jit.script_method
            def forward(self, tensor):
                return self.module(tensor)

        n = 10
        tensor = torch.rand((n, 1), device="cuda")
        factor = 2
        model = MyModule(tensor, factor).cuda()

        jit_out = model(tensor)
        py_out = F.scale(tensor, factor)

        self.assertTrue(torch.allclose(jit_out, py_out, atol=5e-4, rtol=1e-4))

if __name__ == '__main__':
    unittest.main()
