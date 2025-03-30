---
layout: post
---

# Multi-modal 3D Perception

3D perception, including but not limited to semantic segmentation and object detection, commonly utilize sensor redundancy i.e. use LiDAR, RADAR, camera, etc. to capture the same environment in multiple ways to more robustly capture informtaion of the surrounding environment. It's an open research question on how the different modalities should be best fused to process information efficiently and roustly. This blog surveys multi-modal fusion methods for 3D Semantic Segmentation. 

*Last update date: 2023-11-20*

## PointPainting: Sequential Fusion for 3D Object Detection
[[Arxiv]](https://arxiv.org/abs/1911.10150)

* Method: augment object detection by running image-based semantic segmentation and providing the predicted logits to the point cloud as additional channel input
* Main drawback: LiDAR as main modality i.e. if LiDAR fails the whole pipeline fails, sparsity i.e. all non-projected pixels are lost information, and directly passing in semantic logits as input to LiDAR pointcloud is not the most effective method

## DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection
[[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_DeepFusion_Lidar-Camera_Deep_Fusion_for_Multi-Modal_3D_Object_Detection_CVPR_2022_paper.pdf)

* Also a "point painting" method but with improvements
* Camera and LiDAR augmentation handled separately via InverseAug to allow applying data augmentation for each sensor input independently
* Fusion applied at feature level instead of semantics by combining lidar-camera features throgh cross attention
* LiDAR feature used as query and hence allows for more flexible feature fusion as it is not only relying on the back-projected pixel features
* Main drawback: still a lossy information propagation where features from image would be dropped, and dependent on LiDAR for geometric accuracy

## LIF-Seg: LiDAR and Camera Image Fusion for 3D LiDAR Semantic Segmentation
[[Paper]](https://arxiv.org/pdf/2108.07511.pdf)

* Fuse low level and high level features from images to improved LiDAR Segmentation
* Two Stage fusion: 1) coarse feature fusion fuses lower level image features with extracted LiDAR features  via back-projection, then  2) higher level features from an Image Segmetnation Network concatenated to learn offset for finer back-projection, which is used to backproject both coarse and fine context to concatenate with LiDAR for LiDAR Segmentation through UNet
* Main drawback, despite the complexity of the two-stage framework, the performance improvement does not seem as significant

## UniSeg: A Unified Multi-Modal LiDAR Segmentation Network and the OpenPCSeg Codebase
[[Paper]](https://arxiv.org/pdf/2309.05573.pdf)

* Approaches multi-modal fusion by fusing multiple LiDAR reprsentation with image features (augments RPVNet architecture with attention and multi-modal fusion)

* LMA (Learnable Cross-modal Association) fuses range image, voxel features, and 2D feature map via deformable cross-modal attention between (image, range) and (image, voxel) with the LiDAR mode as query (corresponding lidar point/voxel projected to image as reference point and offsets predicted for running atention on fixed set of deformable neighbors)

* After fusing the image features, Learnable crossView Association utilizes similar deformable cross attention module but attends to (points, range image) and (points, voxel) features with points as query. The point-wise range features, point-wise voxel features, and point features are concatenated and agregated via attention-based pooling inspired from [RPVNet](https://arxiv.org/abs/2103.12978).

* Also provides "OpenPCSeg codebase, which is the largest and most comprehensive outdoor LiDAR segmentation codebase" to date of paper's publication.
