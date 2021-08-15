# Pyramid Spatial-Temporal Aggregation for Video-based Person Re-Identification

___

This is repository contains the code for the paper:

Pyramid Spatial-Temporal Aggregation for Video-based Person Re-Identification
Yingquan Wang, Pingping Zhang, Shang Gao, Xia Geng, Hu Lu, Dong Wang
ICCV 2021

## Abstract
Video-based person re-identification aims to associate the video clips of the same person across multiple non-overlapping cameras. Spatial-temporal information provides richer and complementary information between frames, which is crucial to distinguish the target person when occlusion occurs. This paper proposes a novel Pyramid Spatial-Temporal Aggregation (PSTA) framework to aggregate the frame-level features progressively and fuse the hierarchical temporal features into a final video-level representation. Thus, short-term and long-term temporal information could be well exploited by different hierarchies. Furthermore, a Spatial-Temporal Aggregation Module (STAM) is proposed to enhance the aggregation capability of PSTA. It mainly consists of two novel attention blocks: Spatial Reference Attention (SRA) and Temporal Reference Attention (TRA). SRA explores the spatial correlations within a frame to determine the attention weight of each location. While TRA extends SRA with the correlations between adjacent frames, temporal consistency information can be fully explored to suppress the interference features and strengthen the discriminative ones. Extensive experiments on several challenging benchmarks demonstrate the effectiveness of the proposed PSTA, and our full model reaches $91.5\%$ and $98.3\%$ rank-1 accuracy on MARS and DukeMTMC-VideoReID benchmarks.

## Training and Test

```
# For MARS
bash scripts/Mars_train.sh 
bash scripts/Mars_test.sh
```

## Citation 

if you use this code for your research, please cite our paper:

```
@inproceedings{PSTA,
	title = {Pyramid Spatial-Temporal Aggregation for Video-based Person Re-Identification},
	author = {Yingquan Wang and Pingping Zhang and Shang Gao and Xia Geng and Hu Lu and Dong Wang},
	booktitle = {ICCV},
	year = {2021}
}
```



## Platform

This code was developed and tested with pytorch version 1.3.1.

# Acknowledgments

---

This code is based on the implementations of [baseline](https://github.com/yuange250/not_so_strong_baseline_for_video_based_person_reID)
