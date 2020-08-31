# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F

class DWTForward(nn.Module):

    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                          hl[None,::-1,::-1], hh[None,::-1,::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                          hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y

def dwt_init(x):
    # in_batch, in_channel, in_height, in_width = x.size()
    # h_list_1 = [i for i in range(0, in_height, 2)]
    # h_list_2 = [i for i in range(1, in_height, 2)]
    # w_list_1 = [i for i in range(0, in_width, 2)]
    # w_list_2 = [i for i in range(1, in_width, 2)]
    # 奇数行 偶数行
    # x01 = x[:, :, 0::2, :] / 2
    # x02 = x[:, :, 1::2, :] / 2
    # 偶数列 奇数列

    # x1 奇数行 奇数列
    # x2 奇数行 偶数列
    # x3 偶数行 奇数列
    # x4 偶数行 偶数列

    # x1 = x01[:, :, :, 0::2]
    # x2 = x01[:, :, :, 1::2]
    # x3 = x02[:, :, :, 0::2]
    # x4 = x02[:, :, :, 1::2]

    # 系数
    # x_LL = x1 + x2 + x3 + x4
    # x_HL = -x1 - x2 + x3 + x4
    # x_LH = -x1 + x2 - x3 + x4
    # x_HH = x1 - x2 - x3 + x4
    # x_LL = x1 + x2 + x3 + x4
    # x_HL = x1 + x2 - x3 - x4
    # x_LH = x1 - x2 + x3 - x4
    # x_HH = x1 - x2 - x3 + x4
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2

    # sz(1) sz(2)   sz(3)    sz(4)
    # in_batch, in_channel, in_height, in_width = x.size()
    # # sz(1) sz(2)/4 sz(3)*2  sz(4)*2
    # out_batch, out_channel, out_height, out_width = in_batch, int(
    #     in_channel / (r ** 2)), r * in_height, r * in_width
    # # 1:4 5:8 9:12 12:16
    # x1 = x[:, 0:out_channel, :, :] / 2
    # x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    # x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    # x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # h_list_1 = [i for i in range(0, out_height, 2)]
    # h_list_2 = [i for i in range(1, out_height, 2)]
    # w_list_1 = [i for i in range(0, out_width, 2)]
    # w_list_2 = [i for i in range(1, out_width, 2)]

    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    #
    # # x1 奇数行 奇数列
    # # x2 偶数行 奇数列
    # # x3 奇数行 偶数列
    # # x4 偶数行 偶数列
    #
    # h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    # h[:, :, 0::2, 1::2] = x1 - x2 + x3 - x4
    # h[:, :, 1::2, 0::2] = x1 + x2 - x3 - x4
    # h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    # h[:, :, 0::2, 0::2] = x1 + x2 + x3 + x4
    # h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    # h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    # h[:, :, 1::2, 1::2] = x1 - x2 - x3 + x4

    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2



    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4


    return h  # h.reshape(out_batch, out_channel, out_height, out_width)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class prev_block(nn.Module):
    '''conv => BN => ReLU+(conv => BN => ReLU) * 3'''
    def __init__(self, in_ch, out_ch):
        super(prev_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class next_block(nn.Module):
    '''(conv => BN => ReLU) * 3 + conv => BN => ReLU'''

    def __init__(self, in_ch, out_ch):
        super(next_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class level1_prev_block(nn.Module):
    '''conv => ReLU+(conv => BN => ReLU) * 3'''
    def __init__(self, in_ch, out_ch):
        super(level1_prev_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class level1_next_block(nn.Module):
    '''(conv => BN => ReLU) * 3 + conv '''

    def __init__(self, in_ch, out_ch):
        super(level1_next_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class MWCNN(nn.Module):
    def __init__(self):
        super(MWCNN, self).__init__()
        in_channels = 1
        dwt1_num = 4

        level1_num = 160
        level2_num = 256
        level3_num = 256

        # self.DWT = DWT()
        # self.IWT = IWT()



        self.level1_prev = level1_prev_block(dwt1_num, level1_num)
        self.level1_next = level1_next_block(level1_num, dwt1_num)

        self.level2_prev = prev_block(level1_num * 4, level2_num)
        self.level2_next = next_block(level2_num, level1_num * 4)

        self.level3_prev = prev_block(level2_num * 4, level3_num)
        self.level3_next = next_block(level3_num, level2_num * 4)

        self._initialize_weights()
        self.DWT = DWTForward()
        self.IWT = DWTInverse()


    def forward(self, x):
        # 1*4
        dwt1 = self.DWT(x)
        # 4*160
        level1_prev_block = self.level1_prev(dwt1)

        dwt2 = self.DWT(level1_prev_block)
        level2_prev_block = self.level2_prev(dwt2)

        dwt3 = self.DWT(level2_prev_block)
        level3_prev_block = self.level3_prev(dwt3)

        level3_next_block = self.level3_next(level3_prev_block)
        sum3 = torch.add(dwt3, level3_next_block)
        iwt3 = self.IWT(sum3)

        level2_next_block = self.level2_next(iwt3)
        sum2 = torch.add(dwt2, level2_next_block)
        iwt2 = self.IWT(sum2)

        level1_next_block = self.level1_next(iwt2)
        sum1 = torch.add(dwt1, level1_next_block)
        out = self.IWT(sum1)


        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                #print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)