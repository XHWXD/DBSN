#!/bin/bash

logdir=./ckpts/
mkdir -p $logdir

name='rgb_dbsn'
noisetype='poisson_gaussian'
trainset_name='cbsd68'
logname='logger_'${name}"_"${noisetype}".txt"

python rgb_train.py \
--finetune True \
--init_ckpt ./models/rgb_pretrain_sigma_poisson_gaussian.pth \
--trainset $trainset_name \
--log_name $name \
--noise_type $noisetype \
--train_noiseL 40 10 \
--val_noiseL 40 10 \
--middle_channel 96 \
--br1_block_num 8 \
--br2_block_num 8 \
--sigma_mu_middle_channel 32 \
--sigma_mu_output_channel 3 \
--sigma_mu_layers 3 \
--sigma_n_middle_channel 32 \
--sigma_n_output_channel 3 \
--sigma_n_layers 5 \
--activate_fun Relu \
--patch_size 96 \
--batch_size 8 \
--optimizer_type adam \
--lr_policy step \
--lr_dbsn 3e-5 \
--lr_sigma_mu 3e-5 \
--lr_sigma_n 3e-5 \
--epoch 100 \
--decay_rate 0.1 \
--steps "20,40,60,80" \
--log_dir $logdir \
--display_freq 10 \
--save_model_freq 5 
#| tee $logdir/$logname


# #!/bin/bash

# logdir=./ckpts/
# mkdir -p $logdir

# name='rgb_dbsn'
# noisetype='multivariate_gaussian'
# trainset_name='cbsd68'
# logname='logger_'${name}"_"${noisetype}".txt"

# python rgb_train.py \
# --finetune True \
# --resume continue \
# --init_ckpt ./models/rgb_pretrain_sigma_multivariate_gaussian.pth \
# --trainset $trainset_name \
# --log_name $name \
# --noise_type $noisetype \
# --train_noiseL 40 10 \
# --val_noiseL 40 10 \
# --middle_channel 96 \
# --br1_block_num 8 \
# --br2_block_num 8 \
# --sigma_mu_middle_channel 32 \
# --sigma_mu_output_channel 6 \
# --sigma_mu_layers 3 \
# --sigma_n_middle_channel 32 \
# --sigma_n_output_channel 6 \
# --sigma_n_layers 5 \
# --activate_fun Relu \
# --patch_size 96 \
# --batch_size 8 \
# --optimizer_type adam \
# --lr_policy step \
# --lr_dbsn 3e-6 \
# --lr_sigma_mu 3e-6 \
# --lr_sigma_n 3e-6 \
# --epoch 100 \
# --decay_rate 0.1 \
# --steps "20,40,60,80" \
# --log_dir $logdir \
# --display_freq 10 \
# --save_model_freq 5 
# #| tee $logdir/$logname