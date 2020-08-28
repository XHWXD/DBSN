
logdir=./ckpts/
mkdir -p $logdir

name='gray_dbsn'
noisetype='poisson_gaussian'
trainset_name='set12,bsd68,imagenet_val'
logname='logger_'${name}"_"${noisetype}".txt"

python gray_train.py \
--device_ids 0 \
--log_name $name \
--trainset $trainset_name \
--noise_type $noisetype \
--train_noiseL 40 10 \
--val_noiseL 40 10 \
--middle_channel 96 \
--br1_block_num 8 \
--br2_block_num 8 \
--sigma_mu_output_channel 1 \
--sigma_mu_middle_channel 32 \
--sigma_mu_layers 3 \
--sigma_n_output_channel 1 \
--sigma_n_middle_channel 32 \
--sigma_n_layers 5 \
--activate_fun Relu \
--patch_size 96 \
--batch_size 8 \
--optimizer_type adam \
--lr_policy step \
--lr_dbsn 3e-4 \
--lr_sigma_mu 3e-4 \
--lr_sigma_n 3e-4 \
--epoch 100 \
--decay_rate 0.5 \
--steps "20,40,60,70,80,90" \
--log_dir $logdir \
--display_freq 50 \
--save_model_freq 5 \
#| tee $logdir/$logname
