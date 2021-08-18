#!/bin/scripts

python Test.py  --arch 'PSTA'\
                --dataset 'mars'\
                --test_sampler 'Begin_interval'\
                --triplet_distance 'cosine'\
                --test_distance 'cosine'\
                --test_path '/home/wyq/exp/ablation experiment/2021-01-15_22-38-01/rank1_0.91515154_checkpoint_ep350.pth'