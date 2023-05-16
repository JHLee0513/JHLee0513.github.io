---
layout: post
---

# Using SLIC to transfer self-supervised image features to LiDAR Semantic Segmentation

*Last update date: 2023-02-07* <br/>

We roughly follow and implement [Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data
](https://arxiv.org/pdf/2203.16258.pdf) for learning to distill self-supervised image features onto LiDAR for pre-training LiDAR Semantic Segmentation model.

The proposed paper uses SLIC Superpixels ([SLIC Superpixels Compared to State-of-the-art Superpixel Methods](https://core.ac.uk/download/pdf/147983593.pdf)) by Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk to generate regions on input camera image.

## The task of generating Superpixels

The goal of Superpixels to generate clusters given an input image into visually coherent clusters. This allows to view the input image in terms of reigions, rather than the fixed grid of raw pixels. Furthermore, in modern Computer Vision applications the method is often used to generate regions of interest for Depth Estimation, Object Detection, and Semantic Segmentation.

Often explore for region proposal for Object Detection and Semantic Segmentation, methods such as OverFeat[], Multibox[], and DeepMask learn to generate possible regions of interest. In general, SuperPixels refers to more fine grained task of developing clusters such that:

1. Superpixels should adhere well to image boundaries
2. When used to reduce computational complexity as a pre-processing  step, superpixels should be fast to compute, memory efficient, and simple to use.
3. When used for segmentation purposes, superpixels should both increase the speed and improve the quality of the results.

*-Properties of ideal superpiexl as defined in SLIC paper-*

## Evaluation of the Existing SOTA Methods in Superpixel Generation

Existing state-of-the-art (SOTA) methods include GS04, NC05, TP09, QS09, and GC10.

## SLIC: new SOTA method
SLIC, or *simple linear iterative clustering, 'adapts a k-means clustering approach to efficiently generate superpixels.'

The Python/Matlab implementation of SLIC may be found here: [github link)[https://github.com/achanta/SLIC]
<!-- While SSL for DA has been applied effectively to the image and video perception, it has not been explore as much in the 3D vision domain. As an early explorative work, the proposed method utilizes a set of pretext tasks called *Deformation Reconstruction*, augmented with Point Cloud Mixup(PCM) training procedure for domain adaptation on classification and segmentation of point clouds. -->

<!-- ## Unsupervised Domain Adaptation for LiDAR Panoptic Segmentation -->

---
<!-- <p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p> -->
<!-- Remove above link if you don't want to attibute -->