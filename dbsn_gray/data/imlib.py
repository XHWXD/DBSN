# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import h5py
from PIL import Image

img_ext = ('png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'tif', 'TIF')

def is_image(fname):
    return any(fname.endswith(i) for i in img_ext)

def scan(base_path, path=''): # No '/' or '\' at the end of base_path
    images = []
    cur_base_path = base_path if path == '' else os.path.join(base_path, path)
    for d in os.listdir(cur_base_path):
        tmp_path = os.path.join(cur_base_path, d)
        if os.path.isdir(tmp_path):
            images.extend(scan(base_path, os.path.join(path, d)))
        elif is_image(tmp_path):
            images.append(tmp_path)
        else:
            print('Scanning [%s], [%s] is skipped.' % (base_path, tmp_path))
    return images

class imlib():
    def __init__(self, mode='RGB', fmt='CHW', lib='cv2', h5file=None):

        assert mode.upper() in ('RGB', 'L')
        self.mode = mode.upper()

        assert fmt.upper() in ('HWC', 'CHW', 'NHWC', 'NCHW')
        self.fmt = 'CHW' if fmt.upper() in ('CHW', 'NCHW') else 'HWC'

        assert lib.lower() in ('cv2', 'pillow', 'h5')
        self.lib = lib.lower()
        
        if self.lib == 'h5':
            assert h5file is not None
            self.h5file = h5file
        
        self.dtype = np.uint8

        self._imread = getattr(self, '_imread_%s_%s'%(self.lib, self.mode))
        self._imwrite = getattr(self, '_imwrite_%s_%s'%(self.lib, self.mode))

        self._trans_batch = getattr(self, '_trans_batch_%s_%s'
                                    % (self.mode, self.fmt))
        self._trans_image = getattr(self, '_trans_image_%s_%s'
                                    % (self.mode, self.fmt))
        self._trans_back = getattr(self, '_trans_back_%s_%s'
                                    % (self.mode, self.fmt))

    def _imread_cv2_RGB(self, path):
        return cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1]
    def _imread_cv2_L(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img

    def _imread_pillow_RGB(self, path):
        img = Image.open(path)
        im = np.array(img.convert(self.mode))
        img.close()
        return im
    _imread_pillow_L = _imread_pillow_RGB

    # Assume that h5 files save images in CHW format.
    def _imread_h5_RGB(self, index):
        img = np.array(self.h5file[index]).transpose(1,2,0) # CHW->HWC
        if img.shape[2] == 1:
            img = img.repeat(3, 2)
        return img
    def _imread_h5_L(self, index):
        img = np.array(self.h5file[index]).transpose(1,2,0) # CHW->HWC
        if img.shape[2] == 3:
            raise ValueError("Since the format is unknown, please provide gray "
                             "images directly or edit the code (Examples are "
                             "given in the following several lines).")
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # assume a RGB format
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # assume a BGR format
        else:
            img = img.squeeze(2) # HW for gray format
        return img


    def _imwrite_cv2_RGB(self, image, path):
        cv2.imwrite(path, image[..., ::-1])
    def _imwrite_cv2_L(self, image, path):
        cv2.imwrite(path, image)
    def _imwrite_pillow_RGB(self, image, path):
        Image.fromarray(image).save(path)
    _imwrite_pillow_L = _imwrite_pillow_RGB
    # currently, image saving using h5py is not supported
    _imwrite_h5_RGB = lambda image, path: None
    _imwrite_h5_L = lambda image, path: None

    def _trans_batch_RGB_HWC(self, images):
        return np.ascontiguousarray(images)
    def _trans_batch_RGB_CHW(self, images):
        return np.ascontiguousarray(np.transpose(images, (0, 3, 1, 2)))
    def _trans_batch_L_HWC(self, images):
        return np.ascontiguousarray(np.expand_dims(images, 3))
    def _trans_batch_L_CHW(slef, images):
        return np.ascontiguousarray(np.expand_dims(images, 1))

    def _trans_image_RGB_HWC(self, image):
        return np.ascontiguousarray(image)
    def _trans_image_RGB_CHW(self, image):
        return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    def _trans_image_L_HWC(self, image):
        return np.ascontiguousarray(np.expand_dims(image, 2))
    def _trans_image_L_CHW(self, image):
        return np.ascontiguousarray(np.expand_dims(image, 0))

    def _trans_back_RGB_HWC(self, image):
        return image
    def _trans_back_RGB_CHW(self, image):
        return np.transpose(image, (1, 2, 0))
    def _trans_back_L_HWC(self, image):
        return np.squeeze(image, 2)
    def _trans_back_L_CHW(self, image):
        return np.squeeze(image, 0)

    def read(self, paths):
        if isinstance(paths, (list, tuple)):
            images = [self._imread(path) for path in paths]
            return self._trans_batch(np.array(images))
        return self._trans_image(self._imread(paths))

    def back(self, image):
        return self._trans_back(image)

    def write(self, image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._imwrite(self.back(image), path)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    im_rgb_chw_cv2 = imlib('rgb', fmt='chw', lib='cv2')
    im_rgb_hwc_cv2 = imlib('rgb', fmt='hwc', lib='cv2')
    im_rgb_chw_pil = imlib('rgb', fmt='chw', lib='pillow')
    im_rgb_hwc_pil = imlib('rgb', fmt='hwc', lib='pillow')
    im_l_chw_cv2 = imlib('l', fmt='chw', lib='cv2')
    im_l_hwc_cv2 = imlib('l', fmt='hwc', lib='cv2')
    im_l_chw_pil = imlib('l', fmt='chw', lib='pillow')
    im_l_hwc_pil = imlib('l', fmt='hwc', lib='pillow')
    path = '/home/m/data/test/imgs/000000.jpg'

    img_rgb_chw_cv2 = im_rgb_chw_cv2.read(path)
    print(img_rgb_chw_cv2)#.shape)
    plt.imshow(im_rgb_chw_cv2.back(img_rgb_chw_cv2))
    plt.show()
    im_rgb_chw_cv2.write(img_rgb_chw_cv2,
            (path.replace('000000.jpg', 'img_rgb_chw_cv2.jpg')))
    img_rgb_hwc_cv2 = im_rgb_hwc_cv2.read(path)
    print(img_rgb_hwc_cv2)#.shape)
    plt.imshow(im_rgb_hwc_cv2.back(img_rgb_hwc_cv2))
    plt.show()
    im_rgb_hwc_cv2.write(img_rgb_hwc_cv2,
            (path.replace('000000.jpg', 'img_rgb_hwc_cv2.jpg')))
    img_rgb_chw_pil = im_rgb_chw_pil.read(path)
    print(img_rgb_chw_pil)#.shape)
    plt.imshow(im_rgb_chw_pil.back(img_rgb_chw_pil))
    plt.show()
    im_rgb_chw_pil.write(img_rgb_chw_pil,
            (path.replace('000000.jpg', 'img_rgb_chw_pil.jpg')))
    img_rgb_hwc_pil = im_rgb_hwc_pil.read(path)
    print(img_rgb_hwc_pil)#.shape)
    plt.imshow(im_rgb_hwc_pil.back(img_rgb_hwc_pil))
    plt.show()
    im_rgb_hwc_pil.write(img_rgb_hwc_pil,
            (path.replace('000000.jpg', 'img_rgb_hwc_pil.jpg')))
    img_l_chw_cv2 = im_l_chw_cv2.read(path)
    print(img_l_chw_cv2)#.shape)
    plt.imshow(im_l_chw_cv2.back(img_l_chw_cv2))
    plt.show()
    im_l_chw_cv2.write(img_l_chw_cv2,
            (path.replace('000000.jpg', 'img_l_chw_cv2.jpg')))
    img_l_hwc_cv2 = im_l_hwc_cv2.read(path)
    print(img_l_hwc_cv2)#.shape)
    plt.imshow(im_l_hwc_cv2.back(img_l_hwc_cv2))
    plt.show()
    im_l_hwc_cv2.write(img_l_hwc_cv2,
            (path.replace('000000.jpg', 'img_l_hwc_cv2.jpg')))
    img_l_chw_pil = im_l_chw_pil.read(path)
    print(img_l_chw_pil)#.shape)
    plt.imshow(im_l_chw_pil.back(img_l_chw_pil))
    plt.show()
    im_l_chw_pil.write(img_l_chw_pil,
            (path.replace('000000.jpg', 'img_l_chw_pil.jpg')))
    img_l_hwc_pil = im_l_hwc_pil.read(path)
    print(img_l_hwc_pil)#.shape)
    plt.imshow(im_l_hwc_pil.back(img_l_hwc_pil))
    plt.show()
    im_l_hwc_pil.write(img_l_hwc_pil,
            (path.replace('000000.jpg', 'img_l_hwc_pil.jpg')))
