#!/bin/bash

noisetype='poisson_gaussian'

python rgb_test.py \
--last_ckpt ./models/rgb_poisson_gaussian_s40_c10.pth \
--noise_type $noisetype \
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
--gamma 0.9
