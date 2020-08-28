import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, cuda

class TrimmedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        if 'dilation' in kwargs:
            self.dilation = kwargs['dilation']
            kwargs.pop('dilation')
        else:
            self.dilation = 1

        if 'direction' in kwargs:
            self.direction = kwargs['direction']
            kwargs.pop('direction')
        else:
            self.direction = 0

        super(TrimmedConv2d, self).__init__(*args, **kwargs)

        self.slide_winsize = self.weight.shape[2]*self.weight.shape[3]
        self.last_size = torch.zeros(2)
        self.feature_mask=None
        self.mask_ratio=None
        self.weight_mask=None
        self.mask_ratio_dict=dict()
        self.feature_mask_dict=dict()


    def update_mask(self):
        with torch.no_grad():
            self.feature_mask=self.feature_mask_dict[str(self.direction)].to(self.weight.device)
            self.mask_ratio=self.mask_ratio_dict[str(self.direction)].to(self.weight.device)
            self.weight_mask=self.get_weight_mask().to(self.weight.device)

    def get_weight_mask(self,direction=None):
        weight = np.ones((1, 1, self.kernel_size[0], self.kernel_size[1]))
        weight[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
        return torch.tensor(weight.copy(),dtype=torch.float32)

    def update_feature_mask_dict(self,input_h,input_w):
        with torch.no_grad():
            for direct in range(0,1): 
                mask = torch.ones(1, 1, int(input_h), int(input_w))
                weight_mask=self.get_weight_mask(direct)
                (pad_h,pad_w)=self.padding
                pad=torch.nn.ZeroPad2d((pad_w,pad_w,pad_h,pad_h))
                feature_mask = F.conv2d(pad(mask), weight_mask, bias=None, stride=self.stride, dilation=self.dilation, groups=1)
                mask_ratio = self.slide_winsize / (feature_mask + 1e-8)
                # mask_ratio=torch.sqrt(mask_ratio)
                feature_mask = torch.clamp(feature_mask, 0, 1)
                mask_ratio = torch.mul(mask_ratio, feature_mask)
                self.mask_ratio_dict[str(direct)]=mask_ratio
                self.feature_mask_dict[str(direct)]=feature_mask

    def updata_last_size(self,h,w):
        self.last_size.copy_(torch.tensor((h,w),dtype=torch.int32))

    def forward(self, input):
        if (int(self.last_size[0].item()),int(self.last_size[1].item()))!= (int(input.data.shape[2]), int(input.data.shape[3])):
            self.update_feature_mask_dict(input.data.shape[2],input.data.shape[3])
            self.update_mask()
            self.updata_last_size(input.data.shape[2],input.data.shape[3])
        if self.feature_mask is None or self.mask_ratio is None or self.weight_mask is None:
            #self.update_feature_mask_dict()
            self.update_mask()
        #if self.feature_mask.device  != self.weight.device or self.mask_ratio.device != self.weight.device or self.weight_mask.device!=self.weight.device:
        #    with torch.no_grad():
        w=torch.mul(self.weight, self.weight_mask)
        raw_out = F.conv2d(input,w,self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.feature_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        return output


class MaskConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        if 'dilation' in kwargs:
            self.dilation = kwargs['dilation']
            kwargs.pop('dilation')
        else:
            self.dilation = 1

        if 'direction' in kwargs:
            self.direction = kwargs['direction']
            kwargs.pop('direction')
        else:
            self.direction = 0
            
        super(MaskConv2d, self).__init__(*args, **kwargs)
        self.weight_mask = self.get_weight_mask()


    # remove the center position, [1 1 1;1 0 1;1 1 1]
    def get_weight_mask(self):
        weight = np.ones((1, 1, self.kernel_size[0], self.kernel_size[1]))
        weight[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
        return torch.tensor(weight.copy(), dtype=torch.float32)

    def forward(self, input):
        if self.weight_mask.type() != self.weight.type():
            with torch.no_grad():
                self.weight_mask = self.weight_mask.type(self.weight.type())
        w=torch.mul(self.weight,self.weight_mask)
        output = F.conv2d(input, w, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
        return output
