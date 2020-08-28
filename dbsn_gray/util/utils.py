"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import subprocess
import math
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import init
import random
import glob
import re
from skimage.measure.simple_metrics import compare_psnr

def findLastCheckpoint(save_dir, save_pre):
    file_list = glob.glob(os.path.join(save_dir, save_pre + '*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*" + save_pre +"(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def init_func(m, init_type='kaiming',init_gain=0.02):  # define the initialization function
	classname = m.__class__.__name__
	if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
		if init_type == 'normal':
			init.normal_(m.weight.data, 0.0, init_gain)
		elif init_type == 'xavier':
			init.xavier_normal_(m.weight.data, gain=init_gain)
		elif init_type == 'kaiming':
			init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
		elif init_type == 'orthogonal':
			init.orthogonal_(m.weight.data, gain=init_gain)
		else:
			raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
		if hasattr(m, 'bias') and m.bias is not None:
			init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
		init.normal_(m.weight.data, 1.0, init_gain)
		init.constant_(m.bias.data, 0.0)


# Taken from CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

# [...,h,w] torch/numpy
def rand_crop(img, h, w=None):
    w = h if w is None else w
    hh, ww = img.shape[-2:]
    assert hh >= h and ww >= w
    top, left = random.randint(0, hh - h), random.randint(0, ww - w)
    return img[..., top : top + h, left : left + w]

def weights_init_kaiming(lyr):
	r"""Initializes weights of the model according to the "He" initialization
	method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
	This function is to be called by the torch.nn.Module.apply() method,
	which applies weights_init_kaiming() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		lyr.weight.data = nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
			clamp_(-0.025, 0.025)
		nn.init.constant_(lyr.bias.data, 0.0)
        

def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]

def data_augmentation(image, mode):
	r"""Performs dat augmentation of the input image

	Args:
		image: a cv2 (OpenCV) image
		mode: int. Choice of transformation to apply to the image
			0 - no transformation
			1 - flip up and down
			2 - rotate counterwise 90 degree
			3 - rotate 90 degree and flip up and down
			4 - rotate 180 degree
			5 - rotate 180 degree and flip
			6 - rotate 270 degree
			7 - rotate 270 degree and flip
	"""
	out = np.transpose(image, (1, 2, 0))
	if mode == 0:
		# original
		out = out
	elif mode == 1:
		# flip up and down
		out = np.flipud(out)
	elif mode == 2:
		# rotate counterwise 90 degree
		out = np.rot90(out)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		out = np.rot90(out)
		out = np.flipud(out)
	elif mode == 4:
		# rotate 180 degree
		out = np.rot90(out, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		out = np.rot90(out, k=2)
		out = np.flipud(out)
	elif mode == 6:
		# rotate 270 degree
		out = np.rot90(out, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.rot90(out, k=3)
		out = np.flipud(out)
	else:
		raise Exception('Invalid choice of image transformation')
	return np.transpose(out, (2, 0, 1))

def variable_to_cv2_image(varim):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		varim: a torch.autograd.Variable
	"""
	nchannels = varim.size()[1]
	if nchannels == 1:
		res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		res = varim.data.cpu().numpy()[0]
		res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
		res = (res*255.).clip(0, 255).astype(np.uint8)
	else:
		raise Exception('Number of color channels not supported')
	return res

def init_logger(argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(argdict.log_dir, 'log.txt'), mode='a')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	logger.info("Arguments: ")
	for k in argdict.__dict__:
		logger.info("\t{}: {}".format(k, argdict.__dict__[k]))

	return logger

def init_logger_ipol(result_dir='',logfile='out.txt'):
	r"""Initializes a logging.Logger in order to log the results after
	testing a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join
	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, logfile), mode='w')
	formatter = logging.Formatter('%(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='a')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)

def svd_orthogonalization(lyr):
	r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "FFDNet:	Toward a fast
	and flexible solution for CNN based image denoising." Zhang et al. (2017).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		# Convert filter matrix to numpy array
		weights = weights.cpu().numpy()

		# SVD decomposition and orthogonalization
		mat_u, _, mat_vh = np.linalg.svd(weights, full_matrices=False)
		weights = np.dot(mat_u, mat_vh)

		# As full_matrices=False we don't need to set s[:] = 1 and do mat_u*s
		lyr.weight.data = torch.Tensor(weights).view(f1, f2, c_in, c_out).\
			permute(3, 2, 0, 1).type(dtype)
	else:
		pass

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary

	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict

def is_rgb(im_path):
	r""" Returns True if the image in im_path is an RGB image
	"""
	from skimage.io import imread
	rgb = False
	im = imread(im_path)
	if (len(im.shape) == 3):
		if not(np.allclose(im[...,0], im[...,1]) and np.allclose(im[...,2], im[...,1])):
			rgb = True
	print("rgb: {}".format(rgb))
	print("im shape: {}".format(im.shape))
	return rgb

def tensor_left_rot90(img,k=1,axes=(0,1),detach=False):
	if detach:
		out=img.clone().detach()
	else:
		out=img.clone()
	k=k%4
	if k==1:
		out = out.transpose(dim0=axes[0],dim1=axes[1]).flip(axes[0])
	elif k==2:
		out = out.flip(axes[0]).flip(axes[1])
	elif k==3:
		out = out.transpose(dim0=axes[0],dim1=axes[1]).flip(axes[1])
	return out

def tensor_right_rot90(img,k=1,axes=(0,1),detach=False):
	if detach:
		out = img.clone().detach()
	else:
		out = img.clone()
	k=k%4
	if k==1:
		out = out.transpose(dim0=axes[0],dim1=axes[1]).flip(axes[1])
	elif k==2:
		out = out.flip(axes[0]).flip(axes[1])
	elif k==3:
		out = out.transpose(dim0=axes[0],dim1=axes[1]).flip(axes[0])
	return out

def img_set_direction(img,direction,axes=(0,1),detach=True):
	if direction >=0 and direction <4:
		out = tensor_left_rot90(img, k=direction, axes=axes,detach=detach)
	elif direction>=4 and direction <8:
		out = img.flip(axes[1])
		out = tensor_left_rot90(out, k=direction - 4, axes=axes,detach=detach)

	elif direction<=0 and direction >-4:
		out=tensor_right_rot90(img,k=-direction,axes=axes,detach=detach)
	elif direction<=-4 and direction >-8:
		out=tensor_right_rot90(img,k=(-direction-4),axes=axes,detach=detach)
		out =out.flip(axes[1])
	return out

def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)