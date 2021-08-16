#!/bin/scripts

python Test.py  --arch 'PSTA'\
                --dataset 'duke'\
                --train_sampler 'Random_interval'\
                --test_sampler 'dense'\
                --triplet_distance 'cosine'\
                --test_distance 'cosine'\
                --print_performance True\
