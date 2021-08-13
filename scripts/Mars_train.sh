#!/bin/scripts

python Train.py   --arch 'STAM'\
                  --dataset 'mars'\
                  --model_spatial_pool 'avg'\
                  --model_temporal_pool 'avg'\
                  --train_sampler 'Random_interval'\
                  --test_sampler 'Begin_interval'\
                  --transform_method 'consecutive'\
                  --sampler_method 'random'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\
                  --is_cat 'yes'\
                  --feature_method 'cat'\
                  --is_mutual_channel_attention 'yes'\
                  --is_mutual_spatial_attention 'yes'\
                  --is_appearance_channel_attention 'yes'\
                  --is_appearance_spatial_attention 'yes'\
                  --layer_num 3 \
                  --seq_len 8 \
                  --is_down_channel 'yes'\
                  --sampler 'RandomIdentitySampler'\



