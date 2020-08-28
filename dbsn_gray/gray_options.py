import os
import argparse
import datetime
import torch
from pathlib import Path

def str2bool(s):
    """ Can use type=str2bool in the parser.add_argument function """
    return s.lower() in ('t', 'true', 'y', 'yes', '1', 'sure')

parser = argparse.ArgumentParser(description="DBSN_gray")
parser.add_argument("--log_name", type=str, default="dbsn_gray", help="file name for save")
parser.add_argument("--noise_type", type=str, default="gaussian", help="set the noise type")
# data
parser.add_argument("--dataroot", type=str, default="", help="path to set")
parser.add_argument("--trainset", type=str, default="set12,bsd68,imagenet_val", help="training set name")
parser.add_argument("--valset", type=str, default="bsd68", help="validation set name")
#
parser.add_argument("--train_noiseL", type=float, default=[25], nargs="+", help='noise level used on training set')
parser.add_argument("--val_noiseL", type=float, default=[25], nargs="+", help='noise level used on validation set')
#
parser.add_argument("--patch_size", type=int, default=96, help="the patch size of input")
parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
parser.add_argument("--load_thread", type=int, default=0, help="thread for data loader")    # Ming: maybe 0 is not a good choice
# DTCN
parser.add_argument("--input_channel",type=int,default=1,help="the input channel")
parser.add_argument("--output_channel",type=int,default=1,help="the output channel")
parser.add_argument("--middle_channel",type=int,default=96,help="the middle channel")
parser.add_argument("--blindspot_conv_type",type=str,default='Mask',choices=['Trimmed','Mask'], help="type of conv(Trimmed | Mask)")
parser.add_argument("--blindspot_conv_bias",type=str2bool,default=True,help="if blindspot conv need bias")
# branch1
parser.add_argument("--br1_block_num",type=int,default=8,help="the number of dilated conv for branch1")
parser.add_argument("--br1_blindspot_conv_ks",type=int,default=3,help="the basic kernel size of dilated conv")
# branch2
parser.add_argument("--br2_block_num",type=int,default=8,help="the number of dilated conv for branch2")
parser.add_argument("--br2_blindspot_conv_ks",type=int,default=5,help="the basic kernel size of dilated conv")
# net_mu
parser.add_argument("--activate_fun", type=str, default='Relu', choices=['LeakyRelu','Relu'],
                    help='type of activate funcition(LeakyRelu | Relu)')
# net_sigma_mu
parser.add_argument("--sigma_mu_input_channel",type=int,default=0,help="the input channel of net_sigma_mu, NO USE!")
parser.add_argument("--sigma_mu_output_channel",type=int,default=1,help="the output channel of net_sigma_mu")
parser.add_argument("--sigma_mu_middle_channel",type=int,default=16,help="the middle channel of net_sigma_mu")
parser.add_argument("--sigma_mu_layers",type=int,default=3,help="the number of conv in net_sigma_mu")
parser.add_argument("--sigma_mu_kernel_size",type=int,default=1,help="the kernel size of conv in net_sigma_mu")
parser.add_argument("--sigma_mu_bias",type=str2bool,default=True,help="if conv in net_sigma_mu need bias")
# net_sigma_n
parser.add_argument("--sigma_n_input_channel",type=int,default=1,help="the input channel of net_sigma_n")
parser.add_argument("--sigma_n_output_channel",type=int,default=1,help="the output channel of net_sigma_n")
parser.add_argument("--sigma_n_middle_channel",type=int,default=16,help="the middle channel of net_sigma_n")
parser.add_argument("--sigma_n_layers",type=int,default=5,help="the number of conv in net_sigma_n")
parser.add_argument("--sigma_n_kernel_size",type=int,default=1,help="the kernel size of conv in net_sigma_n")
parser.add_argument("--sigma_n_bias",type=str2bool,default=True,help="if conv in net_sigma_n need bias")
# save
parser.add_argument("--init_ckpt",type=str,default="None",help="for pre-training the sub-nets: mu, sigma_mu, sigma_n")
parser.add_argument("--last_ckpt",type=str,default="None",help="the ckpt of last net for checkpoint")
parser.add_argument("--resume", type=str, choices=("continue", "new"), default="new",help="continue to train model")
parser.add_argument("--log_dir", type=str, default='./ckpts/', help='path of log files')
parser.add_argument("--display_freq", type=int, default=50, help="frequency of showing training results on screen")
parser.add_argument("--save_model_freq", type=int, default=5, help="Number of training epchs to save state")
# Training parameters
parser.add_argument("--optimizer_type", type=str, default='Adam', help="the default optimizer")
parser.add_argument("--lr_policy", type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument("--lr_dbsn", type=float, default=3e-4, help="the initial learning rate of backbone network")
parser.add_argument("--lr_sigma_mu", type=float, default=3e-4, help="the initial learning rate of net_sigma_mu")
parser.add_argument("--lr_sigma_n", type=float, default=3e-4, help="the initial learning rate of net_sigma_n")
parser.add_argument("--decay_rate", type=float, default=0.1, help="the decay rate of lr rate")
parser.add_argument("--epoch", type=int, default=90, help="number of epochs the model needs to run")
parser.add_argument("--steps", type=str, default="30,60,70,80", help="schedule steps,use comma(,) between numbers")
# additional
parser.add_argument("--gamma",type=float,default=1,help="additional parameter for MAP Inference, can be set in [0.9,1] (test only)")
# data processing when loaded
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
parser.add_argument('--shuffle', type=str2bool, default=True, help='if true shuffle the data')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                    help='Maximum number of samples allowed per dataset. If the dataset directory contains '
                         'more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--isTrain', type=str2bool, default=True, help='training flag')
parser.add_argument('--mode', type=str, default='L', choices=['L', 'RGB'])
parser.add_argument('--preload', type=str2bool, default=True)
parser.add_argument('--multi_imreader', type=str2bool, default=True)
parser.add_argument('--imlib', type=str, default='cv2', choices=['cv2', 'pillow', 'h5'])
# GPU
parser.add_argument('--device_ids', type=str, default='all', help="integers seperated by comma for selected GPUs, -1 for CPU mode.")
# Option parsing
opt = parser.parse_args()

#
opt.save_prefix = opt.log_name + '_' + opt.noise_type

# parse steps
steps = opt.steps
steps = steps.split(',')
opt.steps = [int(eval(step)) for step in steps]
# parse trainset
trainsets = opt.trainset
if trainsets.find(',') == -1:
    opt.trainset = [trainsets]
else:
    trainsets = trainsets.split(',')
    opt.trainset = [str(trainset).replace(' ','') for trainset in trainsets]
# parse trainset
valsets = opt.valset
if valsets.find(',') == -1:
    opt.valset = [valsets]
else:
    valsets = valsets.split(',')
    opt.valset = [str(valset).replace(' ','') for valset in valsets]


# set gpu ids
cuda_device_count = torch.cuda.device_count()
if opt.device_ids == 'all':
    # GT 710 (3.5), GT 610 (2.1)
    device_ids = [i for i in range(cuda_device_count)]
else:
    device_ids = [int(i) for i in opt.device_ids.split(',') if int(i) >= 0]
opt.device_ids = [i for i in device_ids if torch.cuda.get_device_capability(i) >= (4,0)]
if len(opt.device_ids) == 0 and len(device_ids) > 0:
    opt.device_ids = device_ids
    print('You\'re using GPUs with computing capability < 4')
elif len(opt.device_ids) != len(device_ids):
    print('GPUs with computing capability < 4 have been disabled')

if len(opt.device_ids) > 0:
    assert torch.cuda.is_available(), 'No cuda available !!!'
    torch.cuda.set_device(opt.device_ids[0])
    print('The GPUs you are using:')
    for gpu_id in opt.device_ids:
        print(' %2d *%s* with capability %d.%d' % (gpu_id,
                torch.cuda.get_device_name(gpu_id),
                *torch.cuda.get_device_capability(gpu_id)))
else:
    print('You are using CPU mode')

# print('\tParameteres list:')
# for key in opt.__dict__:
#     print('\t'+key+': '+str(opt.__dict__[key]))
