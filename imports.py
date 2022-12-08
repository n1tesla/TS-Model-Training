import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import random
import os
import sys
import warnings
from warnings import warn
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import fastcore
from fastcore.imports import *
from fastcore.basics import *
from fastcore.xtras import *
from fastcore.test import *
from fastcore.foundation import *
from fastcore.meta import *
from fastcore.dispatch import *

import fastai
from fastai.basics import *
from fastai.imports import *
from fastai.torch_core import *
from fastai.callback.tracker import EarlyStoppingCallback,ReduceLROnPlateau

import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpus = defaults.cpus

def get_gpu_memory():
    import subprocess
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_values = [round(int(x.split()[0])/1024, 2) for i, x in enumerate(memory_info)]
    return memory_values

def get_ram_memory():
    nbytes = psutil.virtual_memory().total
    return round(nbytes / 1024**3, 2)

class Timer:
    def start(self, verbose=True):
        self.all_elapsed = 0
        self.n = 0
        self.verbose = verbose
        self.start_dt = datetime.datetime.now()
        self.start_dt0 = self.start_dt

    def elapsed(self):
        end_dt = datetime.datetime.now()
        self.n += 1
        assert hasattr(self, "start_dt0"), "You need to first use timer.start()"
        elapsed = end_dt - self.start_dt
        if self.all_elapsed == 0: self.all_elapsed = elapsed
        else: self.all_elapsed += elapsed
        pv(f'Elapsed time ({self.n:3}): {elapsed}', self.verbose)
        self.start_dt = datetime.datetime.now()
        if not self.verbose:
            return elapsed

    def stop(self):
        end_dt = datetime.datetime.now()
        self.n += 1
        assert hasattr(self, "start_dt0"), "You need to first use timer.start()"
        elapsed = end_dt - self.start_dt
        if self.all_elapsed == 0: self.all_elapsed = elapsed
        else: self.all_elapsed += elapsed
        total_elapsed = end_dt - self.start_dt0
        delattr(self, "start_dt0")
        delattr(self, "start_dt")
        if self.verbose:
            if self.n > 1:
                print(f'Elapsed time ({self.n:3}): {elapsed}')
                print(f'Total time        : {self.all_elapsed}')
            else:
                print(f'Total time        : {total_elapsed}')
        else: return total_elapsed

timer = Timer()

def my_setup(*pkgs):
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import platform
        print(f'os             : {platform.platform()}')
    except:
        pass
    try:
        from platform import python_version
        print(f'python         : {python_version()}')
    except:
        pass
    try:
        import tsai
        print(f'tsai           : {tsai.__version__}')
    except:
        print(f'tsai           : N/A')
    try:
        import fastai
        print(f'fastai         : {fastai.__version__}')
    except:
        print(f'fastai         : N/A')
    try:
        import fastcore
        print(f'fastcore       : {fastcore.__version__}')
    except:
        print(f'fastcore       : N/A')

    if pkgs:
        for pkg in pkgs:
            try:
                print(f'{pkg.__name__:15}: {pkg.__version__}')
            except:
                pass
    try:
        import torch
        print(f'torch          : {torch.__version__}')
        try:
            import torch_xla
            print(f'device         : TPU')
        except:
            iscuda = torch.cuda.is_available()
            if iscuda:
                device_count = torch.cuda.device_count()
                gpu_text = 'gpu' if device_count == 1 else 'gpus'
                print(
                    f'device         : {device_count} {gpu_text} ({[torch.cuda.get_device_name(i) for i in range(device_count)]})')
            else:
                print(f'device         : {device}')
    except:
        pass
    try:
        print(f'cpu cores      : {cpus}')
    except:
        print(f'cpu cores      : N/A')
    try:
        print(f'RAM            : {get_ram_memory()} GB')
    except:
        print(f'RAM            : N/A')
    try:
        print(f'GPU memory     : {get_gpu_memory()} GB')
    except:
        print(f'GPU memory     : N/A')

computer_setup = my_setup
