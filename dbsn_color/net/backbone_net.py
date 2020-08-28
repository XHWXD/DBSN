import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blind_spot_conv import BlindSpotConv
from util.utils import init_weights, weights_init_kaiming
from functools import partial


class Inception_block(nn.Module):
    def __init__(self, inplanes, kernel_size, dilation, bias, activate_fun):
        super(Inception_block, self).__init__()
        #
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        #
        pad_size = (kernel_size+(kernel_size-1)*(dilation-1)-1)//2
        # inception_br1 ----------------------------------------------
        lyr_br1=[]
        # 1x1 conv
        lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br1.append(self.relu())
        # # case1: two 3x3 dilated-conv
        # lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        # lyr_br1.append(self.relu())
        # lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        # lyr_br1.append(self.relu())
        # case2: one 5x5 dilated-conv
        tmp_kernel_size = 5
        tmp_pad_size = (tmp_kernel_size+(tmp_kernel_size-1)*(dilation-1)-1)//2
        lyr_br1.append(nn.Conv2d(inplanes, inplanes, kernel_size=tmp_kernel_size, padding=tmp_pad_size, dilation=dilation, bias=bias))
        lyr_br1.append(self.relu())
        self.inception_br1=nn.Sequential(*lyr_br1)
        init_weights(self.inception_br1)
        #
        # inception_br2 ----------------------------------------------
        lyr_br2=[]
        # 1x1 conv
        lyr_br2.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br2.append(self.relu())
        # 3x3 dilated-conv
        lyr_br2.append(nn.Conv2d(inplanes, inplanes, kernel_size, padding=pad_size, dilation=dilation, bias=bias))
        lyr_br2.append(self.relu())
        self.inception_br2=nn.Sequential(*lyr_br2)
        init_weights(self.inception_br2)
        #
        # inception_br3 ----------------------------------------------
        lyr_br3=[]
        # 1x1 conv
        lyr_br3.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias))
        lyr_br3.append(self.relu())
        self.inception_br3=nn.Sequential(*lyr_br3)
        init_weights(self.inception_br3)
        # Concat three inception branches
        self.concat = nn.Conv2d(inplanes*3,inplanes,kernel_size=1,bias=bias)
        self.concat.apply(weights_init_kaiming)
        # 1x1 convs
        lyr=[]
        lyr.append(nn.Conv2d(inplanes,inplanes,kernel_size=1,bias=bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(inplanes,inplanes,kernel_size=1,bias=bias))
        lyr.append(self.relu())
        self.middle_1x1_convs=nn.Sequential(*lyr)
        init_weights(self.middle_1x1_convs)
  

    def forward(self, x):
        residual = x
        x1 = self.inception_br1(x)
        x2 = self.inception_br2(x)
        x3 = self.inception_br3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.concat(out)
        out = torch.relu_(out)
        out = out + residual
        out = self.middle_1x1_convs(out)
        return out


class DBSN_branch(nn.Module):
    def __init__(self, inplanes, bs_conv_type, bs_conv_bias, bs_conv_ks, block_num, activate_fun):
        super(DBSN_branch, self).__init__()
        # 
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        #
        dilation_base=(bs_conv_ks+1)//2
        #
        lyr=[]
        lyr.append(BlindSpotConv(inplanes, inplanes, bs_conv_ks, stride=1, dilation=1, bias=bs_conv_bias, conv_type=bs_conv_type))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        lyr.append(self.relu())
        #
        for i in range(block_num):
            lyr.append(Inception_block(inplanes, kernel_size=3, dilation=dilation_base, bias=bs_conv_bias, activate_fun=activate_fun))
        #
        lyr.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bs_conv_bias))
        self.branch=nn.Sequential(*lyr)
        init_weights(self.branch)

    def forward(self,x):
        return self.branch(x)

class DBSN_Model(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, 
                blindspot_conv_type, blindspot_conv_bias,
                br1_blindspot_conv_ks, br1_block_num, 
                br2_blindspot_conv_ks, br2_block_num,
                activate_fun):
        super(DBSN_Model,self).__init__()
        #
        if activate_fun == 'Relu':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = partial(nn.ReLU, inplace=True)
        elif activate_fun == 'LeakyRelu':
            # self.relu = nn.LeakyReLU(0.1)
            self.relu = partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            raise ValueError('activate_fun [%s] is not found.' % (activate_fun))
        # Head of DBSN
        lyr = []
        lyr.append(nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=blindspot_conv_bias))
        lyr.append(self.relu())
        self.dbsn_head = nn.Sequential(*lyr)
        init_weights(self.dbsn_head)

        self.br1 = DBSN_branch(mid_ch, blindspot_conv_type, blindspot_conv_bias, br1_blindspot_conv_ks, br1_block_num, activate_fun)
        self.br2 = DBSN_branch(mid_ch, blindspot_conv_type, blindspot_conv_bias, br2_blindspot_conv_ks, br2_block_num, activate_fun)

        # Concat two branches
        self.concat = nn.Conv2d(mid_ch*2,mid_ch,kernel_size=1,bias=blindspot_conv_bias)
        self.concat.apply(weights_init_kaiming)
        # 1x1 convs
        lyr=[]
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=blindspot_conv_bias))
        lyr.append(self.relu())
        lyr.append(nn.Conv2d(mid_ch,out_ch,kernel_size=1,bias=blindspot_conv_bias))
        self.dbsn_tail=nn.Sequential(*lyr)
        init_weights(self.dbsn_tail)

    def forward(self, x):
        x = self.dbsn_head(x)
        x1 = self.br1(x)     
        x2 = self.br2(x)
        x_concat = torch.cat((x1,x2), dim=1)
        x = self.concat(x_concat)
        return self.dbsn_tail(x), x

