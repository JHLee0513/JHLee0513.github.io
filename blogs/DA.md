---
layout: post
---

# Unsupervised LiDAR Domain Adaptation

This blog post is a survey on unsupervised LiDAR Domain Adaptation, particularly for Deep Learning based 3D Segmentation. Please note that each section description is a mix of paraphrases from the paper as well as my own words.

*Last update date: 2023-02-01*

## Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds
<a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Yi_Complete__Label_A_Domain_Adaptation_Approach_to_Semantic_Segmentation_CVPR_2021_paper.pdf">Paper Link</a>
*venue: CVPR 2021*
*Authors: Li Yi, Boqing Gong, Thomas Funkhouser* <br>

In C&L, the authors propose a framework that reduces the domain gap by utilizing reconsructed point clouds as the canocical domain for semantic segmentation. Specifically, they design a Sparse Voxel Completion Network (SVCN) to complete the 3D surfaces of a sparse point cloud. Notably, SVCN can be trained via self-supervision for each domain. Given a sparse point cloud as input, SVCN generates a completed point cloud that is fed to the segmentation network. Since a canonical domain is used, the segmentation network is shared across domains. 

## Domain Transfer for Semantic Segmentation of LiDAR Data using Deep Neural Networks
<a href="https://ras.papercept.net/images/temp/IROS/files/0060.pdf">Paper Link</a>
*venue: IROS 2020*
*Authors: Ferdinand Langer, Andres Milioto, Alexandre Haag, Jens Behley, Cyrill Stachniss* <br>

In their paper, the authors propose a novel method towards improving generalization cababilities of a LiDAR segmentation network for sensor-to-sensor domain shift. The proposed framework involves using aggregated LiDAR scans to generate scans from one domain that matches the characteristics of another, such that the trained segmentation network on source domain generalizes to the target domain. With a few filtering logic and geodesic correlation alignment during training, the trained segmentation network can generalize to the target domain in comparison to the baseline.

## mDALU: Multi-Source Domain Adaptation and Label Unification with Partial Datasets
<a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Gong_mDALU_Multi-Source_Domain_Adaptation_and_Label_Unification_With_Partial_Datasets_ICCV_2021_paper.pdf">Paper Link</a>
*venue: ICCV 2021*
*Authors: Rui Gong, Dengxin Dai, Yuhua Chen, Wen Li, Luc Van Gool* <br>


## Cross-modal Learning for Domain Adaptation in 3D Semantic Segmentation

## ConDA: Unsupervised Domain Adaptation for LiDAR Segmentation via Regualarized Domain Concatenation

## SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud

## LidarNet: A Boundary-Aware Domain Adaptation Model for Point Cloud Semantic Segmentation

## ePointDA: An End-to-End Simulation-to-Real Domain Adaptation Framework for LiDAR Point Cloud Segmentation
<a href="https://ojs.aaai.org/index.php/AAAI/article/view/16464/16271">Paper Link</a>

*venue: AAAI-21*
*Authors: Sicheng Zhao, Yezhen Wang23, Bo Li, Bichen Wu, Yang Gao, Pengfei Xu,Trevor Darrell, Kurt Keutzer* <br>

ePointDA focuses on Simluation to Real Domain Adaptation (SRDA) for LiDAR segmentation, with the motivation that annotated synthetic data is easier and efficient to generate with less limitations than that of annotating real life data. Specifically, ePointDA is an end-to-end framework with three modules: self-supervised dropout noise rendering, statistics-invariant and spatially-adaptive feature alignment, and transferable segmentation learning. The combination of three modules are what the authors propose to be necessary components for jointly reducing the domain shift from synthetic to real data, and experiment their framework with a modified SqueezeSegV2 network. 

## Unsupervised Domain Adaptation in LiDAR Semantic Segmentation with Self-Supervision and Gated Adapters

## Unsupervised Domain Adaptation for Point Cloud Semantic Segmentation via Graph Matching

## Fake it, Mix it, Segment it: Bridging the Domain Gap Between Lidar Sensors

## PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation

<!-- ## Unsupervised Domain Adaptation for LiDAR Panoptic Segmentation -->

<!-- ## PV-RCNN: The Top-Performing LiDAR-only Solutions for 3D Detection / 3D Tracking / Domain Adaptation of Waymo Open Dataset Challeneges -->

## Self-Supervised Learning for Domain Adaptation on Point Clouds


<!-- 1. <a href="/blogs/bev">BEV Perception</a> -->

<!-- 1. Waabi: Survey on Waabi's approach to Self-Driving
2. Tesla: Perception-based FSD
3. Wayve:
4. Aurora:
5. NVIDIA:
6. Argoverse:
7. Argo(CMU): -->

<!-- ## Offroad Autonomy -->
<!-- 1. Traversability Estimation
2.  -->

<!-- ## Integrating Learning and Control -->

<!-- ## Self Supervision -->

<!-- ## Imitation Learning -->
<!-- 1. Maximum Margin Planning
2. Max Ent IRL
3. Imitation Learning -->

<!-- ## Multi-task Vision for Autonomous Driving -->

---
<!-- <p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p> -->
<!-- Remove above link if you don't want to attibute -->