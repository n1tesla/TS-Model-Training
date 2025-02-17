# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/140_models.misc.ipynb (unless otherwise specified).

__all__ = ['InputWrapper', 'ResidualWrapper', 'RecursiveWrapper']

# Cell
from ..imports import *
from .layers import *
from .utils import *

# Cell
class InputWrapper(Module):
    def __init__(self, arch, c_in, c_out, seq_len, new_c_in=None, new_seq_len=None, **kwargs):

        new_c_in = ifnone(new_c_in, c_in)
        new_seq_len = ifnone(new_seq_len, seq_len)
        self.new_shape = c_in != new_c_in or seq_len != new_seq_len
        if self.new_shape:
            layers = []
            if c_in != new_c_in:
                lin = nn.Linear(c_in, new_c_in)
                nn.init.constant_(lin.weight, 0)
                layers += [Transpose(1,2), lin, Transpose(1,2)]
                lin2 = nn.Linear(seq_len, new_seq_len)
                nn.init.constant_(lin2.weight, 0)
                layers += [lin2]
            self.new_shape_fn = nn.Sequential(*layers)
        self.model = build_ts_model(arch, c_in=new_c_in, c_out=c_out, seq_len=new_seq_len, **kwargs)
    def forward(self, x):
        if self.new_shape: x = self.new_shape_fn(x)
        return self.model(x)

# Cell
class ResidualWrapper(Module):
    def __init__(self, model):
        self.model = model

    def forward(self, x):
        return x[..., -1] + self.model(x)

# Cell
class RecursiveWrapper(Module):
    def __init__(self, model, n_steps, anchored=False):
        self.model, self.n_steps, self.anchored = model, n_steps, anchored
    def forward(self, x):
        preds = []
        for _ in range(self.n_steps):
            pred = self.model(x)
            preds.append(pred)
            if x.ndim != pred.ndim: pred = pred[:, np.newaxis]
            x = torch.cat((x if self.anchored else x[..., 1:], pred), -1)
        return torch.cat(preds, -1)