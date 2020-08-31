import os
import cv2
import random
import numpy as np
from . import imlib
from os.path import join
from data.base_dataset import BaseDataset
from sympy import *
from scipy.linalg import orth


class DnDataset(BaseDataset):
    def __init__(self, opt, split, dataset_name, noiseL):
        super(DnDataset, self).__init__(opt, split, dataset_name)
        self.split = split
        self.mode = opt.mode  # RGB, YCrCb or L
        self.noise_type = opt.noise_type
        self.preload = opt.preload
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size
        self.flip = not opt.no_flip
        if len(noiseL) == 1:
            self.noiseL = noiseL[0]
            self.get_noiseL = self._get_noiseL_1
        elif len(noiseL) == 2:
            if self.noise_type.lower=='gaussian':
                self.noiseL = noiseL
                self.get_noiseL = self._get_noiseL_2
            else:
                self.noiseL = noiseL
                self.get_noiseL = self._get_noiseL_1
        elif len(noiseL) == 3:
            self.noiseL_p = [noiseL[0], noiseL[1]]
            self.noiseL_g = [noiseL[0], noiseL[2]]
            self.get_noiseL_p = self._get_noiseL_p
            self.get_noiseL_g = self._get_noiseL_g
        else:
            raise ValueError('noiseL should have one or two or three values')

        self.getimage = self.getimage_read
        self.multi_imreader = opt.multi_imreader

        if split == 'train':
            self._getitem = self._getitem_train
        else:
            self._getitem = self._getitem_test

        self._add_noise = getattr(self, '_add_noise_%s'%opt.noise_type)

    def _get_noiseL_1(self):
        return self.noiseL
    def _get_noiseL_2(self):
        return random.uniform(*self.noiseL)
    def _get_noiseL_p(self):
        return random.uniform(*self.noiseL_p)
    def _get_noiseL_g(self):
        return random.uniform(*self.noiseL_g)

    def load_data(self):
        self.len_data = len(self.names)
        if self.preload:
            if self.multi_imreader:
                read_images(self)
            else:
                self.images = [self.imio.read(p) for p in self.images]
            self.getimage = self.getimage_preload


    def getimage_preload(self, index):
        return self.images[index], self.names[index]

    def getimage_read(self, index):
        return self.imio.read(self.images[index]), self.names[index]


    def _getitem_train(self, index):
        image, f_name = self.getimage(index)
        image = self._crop(image)
        # NOTE: controllable by opt.no_flip
        image = self._augment(image) if self.flip else image
        image = self.float(image)
        return {'clean': image,
                'noisy': self._add_noise(image),
                'fname': f_name}

    def _getitem_test(self, index):
        image, f_name = self.getimage(index)
        image = self.float(image)
        return {'clean': image,
                'noisy': self._add_noise(image),
                'fname': f_name}


    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data

    def _crop(self, image):
        ih, iw = image.shape[-2:] # HW for gray and CHW for RGB
        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)
        return image[..., iy:iy+self.patch_size, ix:ix+self.patch_size]

    def _augment(self, img):
        if random.random() < 0.5:   img = img[:, ::-1, :]
        if random.random() < 0.5:   img = img[:, :, ::-1]
        if random.random() < 0.5:   img = img.transpose(0, 2, 1) # CHW
        return np.ascontiguousarray(img)

    def _add_noise_gaussian(self, img):
        if self.split == 'val':
            np.random.seed(seed=0)
        noise = np.random.normal(0, self.get_noiseL()/255.,
                                img.shape).astype(np.float32)            
        return img + noise 
    def _add_noise_poisson_gaussian(self, img):
        # implemented in paper
        noiseLevel = [v/255. for v in self.get_noiseL()]
        sigma_s = noiseLevel[0]
        sigma_c = noiseLevel[1]
        if self.split == 'val':
            np.random.seed(seed=0)
        n1 = np.random.randn(*img.shape)*sigma_s*img
        if self.split == 'val':
            np.random.seed(seed=0)
        n2 = np.random.randn(*img.shape)*sigma_c
        noise = (n1 + n2).astype(np.float32)
        return img + noise 
    def _add_noise_poisson_gaussian_blind(self, img):
        if self.split == 'val':
            np.random.seed(seed=0)
        sigma_s = [v/255. for v in self.get_noiseL_p()]
        sigma_c = [v/255. for v in self.get_noiseL_g()]
        noiseL = np.sqrt((sigma_s**2)*img+(sigma_c**2))
        noise = (np.random.randn(*img.shape)*noiseL).astype(np.float32)
        return img + noise 
    def _add_noise_multivariate_gaussian(self, img):
        _,H,W=img.shape
        L=75/255
        np.random.seed(0) # to keep the same noiseSigma (matrix)
        
        np.random.seed(0) # to keep the same noiseSigma (matrix)
        U=orth(np.random.rand(3,3))
        if self.split == 'val':
            np.random.seed(seed=0)
            D=np.diag(np.random.rand(3))
            np.random.seed(0)
            U=orth(np.random.rand(3,3))
        else:
            D=np.diag(np.random.rand(3))
            U=orth(np.random.rand(3,3))
        tmp = np.matmul(D, U)
        tmp = np.matmul(U.T,np.matmul(D, U))
        tmp = (L**2)*tmp
        noiseSigma=np.abs(tmp)
        if self.split == 'val':
            np.random.seed(0)
        noise = np.random.multivariate_normal([0,0,0], noiseSigma, (H,W)).astype(np.float32)
        noise = noise.transpose(2, 0, 1)
        return img + noise    

def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    obj.images[i] = obj.imio.read(obj.images[i])

    # for _ in range(3):
    #     try:
    #         obj.images[i] = obj.imio.read(obj.images[i])
    #         failed = False
    #         break
    #     except:
    #         failed = True
    # if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
    # may use `from multiprocessing import Pool` instead, but less efficient and
    # NOTE: `multiprocessing.Pool` will duplicate given object for each process.
    from multiprocessing.dummy import Pool, freeze_support
    from tqdm import tqdm
    print('Starting to load images via multiple imreaders')
    pool = Pool() # use all threads by default
    for _ in tqdm(pool.imap(imreader, iter_obj(obj.len_data, obj)),
                  total=obj.len_data):
        pass
    pool.close()
    pool.join()

if __name__ == '__main__':
    pass
