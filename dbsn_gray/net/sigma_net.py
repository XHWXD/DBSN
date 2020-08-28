#
import torch
import torch.nn as nn
from util.utils import init_weights, weights_init_kaiming


# Net_sigma_mu: keep output>0 !
class Sigma_mu_Net(nn.Module):
    def __init__(self,in_ch,out_ch,mid_ch,layers,kernel_size,bias):
        super(Sigma_mu_Net, self).__init__()
        #
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        #
        self.lyr=[]
        self.lyr.append(nn.Conv2d(in_ch,mid_ch,kernel_size=1,bias=bias))
        self.lyr.append(nn.ReLU(inplace=True))
        for l in range(layers-2):
            self.lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size=1,bias=bias))
            self.lyr.append(nn.ReLU(inplace=True))
        self.lyr.append(nn.Conv2d(mid_ch,out_ch,kernel_size=1,bias=bias))
        self.conv=nn.Sequential(*self.lyr)
        init_weights(self.conv)

    def forward(self,x):
        x = self.conv(x)
        return x


class Sigma_n_Net(nn.Module):
    def __init__(self,in_ch,out_ch,mid_ch,layers,kernel_size,bias):
        super(Sigma_n_Net, self).__init__()
        #
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        #
        if layers == 1:
            self.conv_final=nn.Conv2d(in_ch,out_ch,kernel_size,padding=(kernel_size-1) // 2,bias=bias)
            nn.init.zeros_(self.conv_final.weight)
            nn.init.zeros_(self.conv_final.bias)            
        else:
            self.lyr=[]
            self.lyr.append(nn.Conv2d(in_ch,mid_ch,kernel_size,padding=(kernel_size-1) // 2,bias=bias))
            self.lyr.append(nn.ReLU(inplace=True))
            for l in range(layers-2):
                self.lyr.append(nn.Conv2d(mid_ch,mid_ch,kernel_size,padding=(kernel_size-1) // 2,bias=bias))
                self.lyr.append(nn.ReLU(inplace=True))
            self.conv=nn.Sequential(*self.lyr)
            init_weights(self.conv)
            #
            self.conv_final=nn.Conv2d(mid_ch,out_ch,kernel_size,padding=(kernel_size-1) // 2,bias=bias)
            nn.init.zeros_(self.conv_final.weight)
            nn.init.zeros_(self.conv_final.bias)

    def forward(self,x):
        if self.layers == 1:
            x = self.conv_final(x)
        else:
            x = self.conv(x)
            x = self.conv_final(x)
        return x + (3e-4)
