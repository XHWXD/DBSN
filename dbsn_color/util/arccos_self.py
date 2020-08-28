#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:40:03 2019

@author: xhwu
"""


import math
from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F

eps = 1e-8

class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.acos(input) / 3
        tmp_index_1 = input <= -1 + eps
        out[tmp_index_1] = math.pi / 3
        tmp_index_2 = input >= 1 - eps
        out[tmp_index_2] = 0
        tmp_index_3 = ~(tmp_index_1^tmp_index_2)
        
        ctx.save_for_backward(input, tmp_index_3)

        return out

    @staticmethod
    def backward(ctx, grad_h):

        X, tmp_index_3 = ctx.saved_variables

        d_input = torch.zeros(X.shape, device=X.device)
        d_input[tmp_index_3] = -1/(3*(1-X.double()[tmp_index_3].pow(2)).sqrt().float())
        d_input*=grad_h

        return d_input


class ACOS(nn.Module):
    def __init__(self):
        super(ACOS, self).__init__()
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input):
        return LLTMFunction.apply(input)
    