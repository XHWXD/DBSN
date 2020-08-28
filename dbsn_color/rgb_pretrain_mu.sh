#!/bin/bash

logdir=./ckpts/
mkdir -p $logdir

name='rgb_pretrain_mu'
noisetype='poisson_gaussian'
trainset_name='set14,bsd500,mcmaster,kodak24,imagenet_val'
logname='logger_'${name}"_"${noisetype}".txt"

python rgb_pretrain_mu.py \
--trainset $trainset_name \
--log_name $name \
--noise_type $noisetype \
--train_noiseL 40 10 \
--val_noiseL 40 10 \
--middle_channel 96 \
--br1_block_num 8 \
--br2_block_num 8 \
--activate_fun Relu \
--patch_size 96 \
--batch_size 8 \
--optimizer_type adam \
--lr_policy step \
--lr_dbsn 3e-4 \
--epoch 100 \
--decay_rate 0.1 \
--steps "20,60,70,80,90" \
--log_dir $logdir \
--display_freq 50 \
--save_model_freq 5 
#| tee $logdir/$logname
