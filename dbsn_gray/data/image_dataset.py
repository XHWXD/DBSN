import random
import numpy as np
import os
from os.path import join
from .dn_dataset import DnDataset
from . import imlib

rootlist = {
    'bsd68': [
        './datasets/bsd68',
    ],
    'set12': [
        '/home/xhwu/denoising/datasets/set12',
    ],    
    'cbsd68': [
        './datasets/CBSD68',
    ],
    'bsd500': [
        '/home/xhwu/denoising/datasets/BSD500',
    ],
    'set14': [
        '/home/xhwu/denoising/datasets/SET14',
    ],  
    'kodak24': [
        '/home/xhwu/denoising/datasets/KODAK24'
    ], 
    'mcmaster': [
        '/home/xhwu/denoising/datasets/McMaster'
    ], 
    'wed4744': [
        '/home/xhwu/denoising/datasets/WED4744'
    ],
    'div2k': [
        '/home/xhwu/denoising/datasets/DIV2K_train_HR'
    ],
    'imagenet_val': [
        '/home/xhwu/denoising/datasets/imagenet_val_filter'
    ],
}

class ImageDataset(DnDataset):
    def __init__(self, opt, split, dataset_name, noiseL):
        super(ImageDataset, self).__init__(opt, split, dataset_name, noiseL)
        if self.root == '':
            self.root = []
            for i in range(len(dataset_name)):
                found = False
                single_dataset_name = dataset_name[i]
                for root in rootlist[single_dataset_name]:
                    sub_dirs = os.listdir(root)
                    for i in range(len(sub_dirs)):
                        tmp_path = os.path.join(root, sub_dirs[i])
                        if os.path.isdir(tmp_path):
                            self.root.append(tmp_path)
                            found = True
                    else:
                        self.root.append(root)
                        found = True                        
                if not found:
                    raise ValueError('dataset [%s] is not found.' % (single_dataset_name))

        self.names = []
        self.images = []
        for i in range(len(self.root)):
            single_root= self.root[i]
            tmp_names = imlib.scan(single_root)
            self.names.extend(tmp_names)
            # tmp_images = list(map(lambda x: os.path.join(single_root, x), tmp_names))
            self.images.extend(tmp_names)

        self.imio = imlib.imlib(self.mode, lib=opt.imlib)
        self.load_data()
    
    def float(self, img):
        return img.astype(np.float32) / 255.


if __name__ == '__main__':
    pass