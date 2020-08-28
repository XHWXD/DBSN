#
import os
import random
import datetime
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import create_dataset

from rgb_options import opt
from net.backbone_net import DBSN_Model
from net.sigma_net import Sigma_mu_Net, Sigma_n_Net
from util.utils import batch_psnr


seed=0
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # net architecture
    dbsn_net = DBSN_Model(in_ch = args.input_channel,
                            out_ch = args.output_channel,
                            mid_ch = args.middle_channel,
                            blindspot_conv_type = args.blindspot_conv_type,
                            blindspot_conv_bias = args.blindspot_conv_bias,
                            br1_block_num = args.br1_block_num,
                            br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                            br2_block_num = args.br2_block_num,
                            br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                            activate_fun = args.activate_fun)
    sigma_mu_net = Sigma_mu_Net(in_ch=args.middle_channel,
                    out_ch=args.sigma_mu_output_channel,
                    mid_ch=args.sigma_mu_middle_channel,
                    layers=args.sigma_mu_layers,
                    kernel_size=args.sigma_mu_kernel_size,
                    bias=args.sigma_mu_bias)
    sigma_n_net = Sigma_n_Net(in_ch=args.sigma_n_input_channel,
            out_ch=args.sigma_n_output_channel,
            mid_ch=args.sigma_n_middle_channel,
            layers=args.sigma_n_layers,
            kernel_size=args.sigma_n_kernel_size,
            bias=args.sigma_n_bias)

    # Move to GPU
    dbsn_model = nn.DataParallel(dbsn_net, args.device_ids).cuda()
    sigma_mu_model = nn.DataParallel(sigma_mu_net, args.device_ids).cuda()
    sigma_n_model = nn.DataParallel(sigma_n_net, args.device_ids).cuda()

    tmp_ckpt=torch.load(args.last_ckpt,map_location=torch.device('cuda', args.device_ids[0]))
    # Initialize dbsn_model
    pretrained_dict=tmp_ckpt['state_dict_dbsn']
    model_dict=dbsn_model.state_dict()
    pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    assert(len(pretrained_dict)==len(pretrained_dict_update))
    assert(len(pretrained_dict_update)==len(model_dict))
    model_dict.update(pretrained_dict_update)
    dbsn_model.load_state_dict(model_dict)

    # Initialize sigma_mu_model
    pretrained_dict=tmp_ckpt['state_dict_sigma_mu']
    model_dict=sigma_mu_model.state_dict()
    pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    assert(len(pretrained_dict)==len(pretrained_dict_update))
    assert(len(pretrained_dict_update)==len(model_dict))
    model_dict.update(pretrained_dict_update)
    sigma_mu_model.load_state_dict(model_dict)

    # Initialize sigma_n_model
    pretrained_dict=tmp_ckpt['state_dict_sigma_n']
    model_dict=sigma_n_model.state_dict()
    pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    assert(len(pretrained_dict)==len(pretrained_dict_update))
    assert(len(pretrained_dict_update)==len(model_dict))
    model_dict.update(pretrained_dict_update)
    sigma_n_model.load_state_dict(model_dict)

    # set val set
    val_setname = args.valset
    dataset_val = create_dataset(val_setname, 'val', args).load_data()

    # --------------------------------------------
    # Evaluating
    # --------------------------------------------
    dbsn_model.eval()
    sigma_mu_model.eval()
    sigma_n_model.eval()
    with torch.no_grad():
        psnr_val = 0
        for count, data in enumerate(dataset_val):
            # prepare data
            img_val = data['clean'].cuda()
            img_noise_val = data['noisy'].cuda()
            batch,C,H,W = img_noise_val.shape
            # forward
            mu_out_val, mid_out_val = dbsn_model(img_noise_val)
            sigma_mu_out_val = sigma_mu_model(mid_out_val)
            if args.noise_type == 'poisson_gaussian':
                sigma_n_out_val = sigma_n_model(mu_out_val)
            else:
                sigma_n_out_val = sigma_n_model(img_noise_val)
            #
            if args.noise_type == 'gaussian':
                #
                sigma_mu_out_val = sigma_mu_out_val.repeat(1,3,1,1)
                index = torch.LongTensor([0,4,8]).cuda()
                # sigma_mu
                tmp1 = torch.zeros((batch, H, W, 9), device='cuda')
                tmp1 = tmp1.index_copy_(3, index, sigma_mu_out_val.permute(0,2,3,1))
                L_matrix = tmp1.view((batch,H,W,3,3)) 
                sigma_mu = (L_matrix @ L_matrix.transpose(3,4)) # bxmxnx3x3
                # sigma_n
                noise_est_val = sigma_n_out_val.mean(dim=(2,3), keepdim=True).repeat(1,3,H,W)
                noise_est_val = F.softplus(noise_est_val - 4) + (1e-3)
                tmp2 = torch.zeros((batch, H, W, 9), device='cuda')
                tmp2 = tmp2.index_copy_(3, index, noise_est_val.permute(0,2,3,1))
                P_matrix = tmp2.view((batch,H,W,3,3)) 
                sigma_n = (P_matrix @ P_matrix.transpose(3,4)) # bxmxnx3x3
            elif args.noise_type == 'poisson_gaussian':
                index = torch.LongTensor([0,4,8]).cuda()
                # sigma_mu
                tmp1 = torch.zeros((batch, H, W, 9), device='cuda')
                tmp1 = tmp1.index_copy_(3, index, sigma_mu_out_val.permute(0,2,3,1))
                L_matrix = tmp1.view((batch,H,W,3,3)) 
                sigma_mu = (L_matrix @ L_matrix.transpose(3,4)) # bxmxnx3x3
                # sigma_n
                noise_est = F.softplus(sigma_n_out_val - 4) + (1e-3)
                tmp2 = torch.zeros((batch, H, W, 9), device='cuda')
                tmp2 = tmp2.index_copy_(3, index, noise_est.permute(0,2,3,1))
                P_matrix = tmp2.view((batch,H,W,3,3)) 
                sigma_n = (P_matrix @ P_matrix.transpose(3,4)) # bxmxnx3x3
            elif args.noise_type == 'multivariate_gaussian':
                index = torch.LongTensor([0,1,2,4,5,8]).cuda()
                # sigma_mu
                tmp1 = torch.zeros((batch, H, W, 9), device='cuda')
                tmp1 = tmp1.index_copy_(3, index, sigma_mu_out_val.permute(0,2,3,1))
                L_matrix = tmp1.view((batch,H,W,3,3)) 
                sigma_mu_matrix = (L_matrix @ L_matrix.transpose(3,4)) # bxmxnx3x3
                sigma_mu = sigma_mu_matrix.mean(dim=(1,2), keepdim=True).repeat(1,H,W,1,1)
                # sigma_n
                tmp2 = torch.zeros((batch, H, W, 9), device='cuda')
                tmp2 = tmp2.index_copy_(3, index, sigma_n_out_val.permute(0,2,3,1))
                P_matrix = tmp2.view((batch,H,W,3,3)) 
                sigma_n_matrix = (P_matrix @ P_matrix.transpose(3,4)) # bxmxnx3x3
                sigma_n = sigma_n_matrix.mean(dim=(1,2), keepdim=True).repeat(1,H,W,1,1)
            else:
                assert('Unknown noise type!')
            # compute map_out
            mu_x = mu_out_val.permute(0,2,3,1).unsqueeze(-1) 
            y = img_noise_val.permute(0,2,3,1).unsqueeze(-1)
            Ieps = (1e-6)*torch.eye(3,device='cuda').repeat(batch,H,W,1,1)
            sigma_mu_inv = (sigma_mu+Ieps).inverse()
            sigma_n_inv = (sigma_n+Ieps).inverse()
            term_c1 = (sigma_mu_inv + args.gamma*sigma_n_inv + Ieps).inverse()
            term_c2 = sigma_mu_inv @ mu_x + args.gamma*sigma_n_inv @ y
            out = term_c1 @ term_c2
            out = out.squeeze(-1).permute(0,3,1,2)
            # compute PSNR
            psnr = batch_psnr(out.clamp(0., 1.), img_val, 1.)
            psnr_val+=psnr
            # print
            print("Image[%02d]: psnr_dbsn = %.4f " % (count, psnr))
        psnr_val /= len(dataset_val)
        # print
        print("Avg psnr_dbsn: %.4f \n" % (psnr_val))


if __name__ == "__main__":

    main(opt)

    exit(0)





