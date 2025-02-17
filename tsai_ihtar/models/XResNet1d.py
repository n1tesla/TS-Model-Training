# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/112_models.XResNet1d.ipynb (unless otherwise specified).

__all__ = ['xresnet1d18', 'xresnet1d34', 'xresnet1d50', 'xresnet1d101', 'xresnet1d152', 'xresnet1d18_deep',
           'xresnet1d34_deep', 'xresnet1d50_deep', 'xresnet1d18_deeper', 'xresnet1d34_deeper', 'xresnet1d50_deeper']

# Cell
from fastai.vision.models.xresnet import *
from ..imports import *
from .layers import *
from .utils import *

# Cell
@delegates(ResBlock)
def xresnet1d18 (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet18(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d34 (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet34(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d50 (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet50(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d101 (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet101(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d152 (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet152(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d18_deep (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet18_deep(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d34_deep (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet34_deep(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d50_deep (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet50_deep(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d18_deeper (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet18_deeper(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d34_deeper (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet34_deeper(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)
@delegates(ResBlock)
def xresnet1d50_deeper (c_in, c_out, act=nn.ReLU, **kwargs): return xresnet50_deeper(c_in=c_in, n_out=c_out, act_cls=act, ndim=1, **kwargs)