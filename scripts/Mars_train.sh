#!/bin/scripts

python Train.py   --arch 'PSTA'\
                  --dataset 'mars'\
                  --train_sampler 'Random_interval'\
                  --test_sampler 'Begin_interval'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\



