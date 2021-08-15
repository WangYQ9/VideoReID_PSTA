#!/bin/scripts

python Test.py  --arch 'PSTA'\
                --dataset 'duke'\
                --model_spatial_pool 'avg'\
                --model_temporal_pool 'avg'\
                --train_sampler 'Random_interval'\
                --test_sampler 'dense'\
                --triplet_distance 'cosine'\
                --test_distance 'cosine'\
                --layer_num 3 \
                --seq_len 8 \
                --print_performance True\
