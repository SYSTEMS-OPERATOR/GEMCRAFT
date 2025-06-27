import torch
import torch.nn as nn
import types

from gemcraft.seamless import wrap_tensor, _is_feedforward_block, SeamlessWrapper, replace_nonlinear

# Dummy modules to simulate nonlinear layers
class GeGLU(nn.Module):
    def forward(self, x):
        return x

# Uppercase variant to ensure case-insensitive replacement
class GEGLU(nn.Module):
    def forward(self, x):
        return x

class RMSNorm(nn.Module):
    def forward(self, x):
        return x

class QKNorm(nn.Module):
    def forward(self, x):
        return x


def test_wrap_tensor():
    x = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # shape (1,2,2,1)
    wrapped = wrap_tensor(x)
    assert wrapped.shape == (1, 4, 4, 1)
    # corners reflect wrapping
    assert wrapped[0,0,0,0].item() == 4.0  # last row, last col becomes first
    assert wrapped[0,-1,-1,0].item() == 1.0  # first row/col becomes last


def test_is_feedforward_block():
    block = nn.Sequential(nn.Linear(2,2), nn.ReLU(), nn.Linear(2,2))
    assert _is_feedforward_block(block)
    block2 = nn.Sequential(nn.Linear(2,2))
    assert not _is_feedforward_block(block2)


def test_replace_nonlinear():
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = GeGLU()
            self.b = GEGLU()  # uppercase variant
            self.c = nn.Linear(1,1)
    mod = TestModule()
    counters = {"nonlinear_replacements":0}
    replace_nonlinear(mod, counters, report_changes=False)
    assert isinstance(mod.a, nn.Identity)
    assert isinstance(mod.b, nn.Identity)
    assert counters["nonlinear_replacements"] == 2


def test_seamless_wrapper_shape():
    linear = nn.Linear(2,2)
    wrapper = SeamlessWrapper(linear)
    x = torch.ones(1,3,2)
    out = wrapper(x)
    assert out.shape == x.shape
