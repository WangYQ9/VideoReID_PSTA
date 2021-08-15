#!/bin/scripts

python Test.py  --arch 'PSTA'\
                --dataset 'mars'\
                --model_spatial_pool 'avg'\
                --model_temporal_pool 'avg'\
                --train_sampler 'Begin_interval'\
                --test_sampler 'dense'\
                --triplet_distance 'cosine'\
                --test_distance 'cosine'\
                --layer_num 3 \
                --seq_len 8 \
                --print_performance True\
                --test_path '/home/wyq/exp/ablation experiment/2021-06-17_18-48-36/rank1_0.9141414_checkpoint_ep390.pth'