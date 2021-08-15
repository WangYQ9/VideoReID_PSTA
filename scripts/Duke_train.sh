#!/bin/scripts

python Train.py     --arch 'PSTA'\
                    --dataset 'duke'\
                    --model_spatial_pool 'avg'\
                    --model_temporal_pool 'avg'\
                    --train_sampler 'Random_interval'\
                    --test_sampler 'dense'\
                    --triplet_distance 'cosine'\
                    --test_distance 'cosine'\
                    --seq_len 8 \
