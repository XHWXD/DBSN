#!/bin/bash

logdir=./ckpts/
mkdir -p $logdir

name='gray_kd'
noisetype='gaussian'
trainset_name='div2k,wed4744'
logname='logger_'${name}"_"${noisetype}".txt"

python gray_kd_train.py \
--device_ids 1 \
--log_name $name \
--trainset $trainset_name \
--noise_type $noisetype \
--train_noiseL 25 \
--val_noiseL 25 \
--dbsn_ckpt ./models/gray_gaussian_nL25.pth \
--isPretrain True \
--pretrained_cnn_denoiser_path ./models/mwcnn_gaussian_nL25.pth \
--middle_channel 96 \
--br1_block_num 8 \
--br2_block_num 8 \
--sigma_mu_output_channel 1 \
--sigma_mu_middle_channel 16 \
--sigma_mu_layers 3 \
--sigma_n_output_channel 1 \
--sigma_n_middle_channel 16 \
--sigma_n_layers 5 \
--activate_fun Relu \
--patch_size 160 \
--batch_size 32 \
--optimizer_type adam \
--lr_policy step \
--lr_cnn_denoiser 1e-7 \
--epoch 80 \
--decay_rate 0.1 \
--steps "40,60" \
--log_dir $logdir \
--display_freq 10 \
--save_model_freq 5 \
#| tee $logdir/$logname
