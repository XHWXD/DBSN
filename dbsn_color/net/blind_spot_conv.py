import os
import torch
import torch.nn as nn
from .trimmedconv import TrimmedConv2d, MaskConv2d

# NOTE: Since the API of dilation is set, the padding should consider dilation correspondingly.
def BlindSpotConv(in_planes, out_planes, kernel_size, stride=1, dilation=1, bias=False, conv_type='Trimmed'):
    if conv_type.lower()=='trimmed':
        return TrimmedConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=(kernel_size+(kernel_size-1)*(dilation-1)-1)//2, dilation=dilation, bias=bias, direction=0)
    elif conv_type.lower()=='mask':
        return MaskConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=(kernel_size+(kernel_size-1)*(dilation-1)-1)//2, dilation=dilation, bias=bias, direction=0)
    else:
        raise BaseException("Invalid Conv Type!")