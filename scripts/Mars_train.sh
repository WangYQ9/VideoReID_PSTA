#!/bin/scripts

python Train.py   --arch 'PSTA'\
                  --dataset 'mars'\
                  --model_spatial_pool 'avg'\
                  --model_temporal_pool 'avg'\
                  --train_sampler 'Random_interval'\
                  --test_sampler 'Begin_interval'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\
                  --seq_len 8 \



