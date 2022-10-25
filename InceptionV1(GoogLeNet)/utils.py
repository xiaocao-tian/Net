import datetime
import os
import torch
from pathlib import Path


def data_modified(path=__file__):
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def selectDevice(device, batch_size=None):
    s = f"{data_modified()} torch {torch.__version__}"
    device = str(device).strip().lower().replace("cuda", '')
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), "CUDA unavailable"
    
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch_size:
            assert batch_size % n == 0, "batch is not a GPU multiple"
            
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i==0 else space } CUDA:{d} ({p.name}, {p.total_memory / 1024 **2}MB)\n"
    else:
        s += "CPU\n"
    print(s)
    return torch.device('cuda:0' if cuda else 'cpu')