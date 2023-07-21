---
layout: post
---

# Unsupervised LiDAR Domain Adaptation

This blog post is a survey on unsupervised LiDAR Domain Adaptation, particularly for Deep Learning based 3D Segmentation. For now I have copied and pasted the abstract, and I plan on writing more about my understanding and analysis of each paper in the future.

*Last update date: 2023-02-05*

## [Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds](https://openaccess.thecvf.com/content/CVPR2021/papers/Yi_Complete__Label_A_Domain_Adaptation_Approach_to_Semantic_Segmentation_CVPR_2021_paper.pdf)

*Venue: CVPR 2021*<br>
*Authors: Li Yi, Boqing Gong, Thomas Funkhouser* <br>
*Abstract: We study an unsupervised domain adaptation problem for
the semantic labeling of 3D point clouds, with a particular
focus on domain discrepancies induced by different LiDAR
sensors. Based on the observation that sparse 3D point
clouds are sampled from 3D surfaces, we take a Complete
and Label approach to recover the underlying surfaces before passing them to a segmentation network. Specifically,
we design a Sparse Voxel Completion Network (SVCN) to
complete the 3D surfaces of a sparse point cloud. Unlike
semantic labels, to obtain training pairs for SVCN requires
no manual labeling. We also introduce local adversarial
learning to model the surface prior. The recovered 3D surfaces serve as a canonical domain, from which semantic
labels can transfer across different LiDAR sensors. Experiments and ablation studies with our new benchmark for
cross-domain semantic labeling of LiDAR data show that the
proposed approach provides 6.3-37.6% better performance
than previous domain adaptation methods.*
<!-- In C&L, the authors propose a framework that reduces the domain gap by utilizing reconsructed point clouds as the canocical domain for semantic segmentation. Specifically, they design a Sparse Voxel Completion Network (SVCN) to complete the 3D surfaces of a sparse point cloud. Notably, SVCN can be trained via self-supervision for each domain. Given a sparse point cloud as input, SVCN generates a completed point cloud that is fed to the segmentation network. Since a canonical domain is used, the segmentation network is shared across domains.  -->

## [Domain Transfer for Semantic Segmentation of LiDAR Data using Deep Neural Networks](https://ras.papercept.net/images/temp/IROS/files/0060.pdf)
*Venue: IROS 2020*<br>
*Authors: Ferdinand Langer, Andres Milioto, Alexandre Haag, Jens Behley, Cyrill Stachniss* <br>
*Abstract: Inferring semantic information towards an understanding of the surrounding environment is crucial for
autonomous vehicles to drive safely. Deep learning-based segmentation methods can infer semantic information directly
from laser range data, even in the absence of other sensor
modalities such as cameras. In this paper, we address improving
the generalization capabilities of such deep learning models to
range data that was captured using a different sensor and in
situations where no labeled data is available for the new sensor
setup. Our approach assists the domain transfer of a LiDARonly semantic segmentation model to a different sensor and
environment exploiting existing geometric mapping systems. To
this end, we fuse sequential scans in the source dataset into a
dense mesh and render semi-synthetic scans that match those
of the target sensor setup. Unlike simulation, this approach
provides a real-to-real transfer of geometric information and
delivers additionally more accurate remission information. We
implemented and thoroughly tested our approach by transferring semantic scans between two different real-world datasets
with different sensor setups. Our experiments show that we
can improve the segmentation performance substantially with
zero manual re-labeling. This approach solves the number one
feature request since we relea*
<!-- In their paper, the authors propose a novel method towards improving generalization cababilities of a LiDAR segmentation network for sensor-to-sensor domain shift. The proposed framework involves using aggregated LiDAR scans to generate scans from one domain that matches the characteristics of another, such that the trained segmentation network on source domain generalizes to the target domain. With a few filtering logic and geodesic correlation alignment during training, the trained segmentation network can generalize to the target domain in comparison to the baseline. -->

## [mDALU: Multi-Source Domain Adaptation and Label Unification with Partial Datasets](https://openaccess.thecvf.com/content/ICCV2021/papers/Gong_mDALU_Multi-Source_Domain_Adaptation_and_Label_Unification_With_Partial_Datasets_ICCV_2021_paper.pdf)
*Venue: ICCV 2021*<br>
*Authors: Rui Gong, Dengxin Dai, Yuhua Chen, Wen Li, Luc Van Gool* <br>
*Abstract: One challenge of object recognition is to generalize to
new domains, to more classes and/or to new modalities.
This necessitates methods to combine and reuse existing
datasets that may belong to different domains, have partial annotations, and/or have different data modalities. This
paper formulates this as a multi-source domain adaptation and label unification problem, and proposes a novel
method for it. Our method consists of a partially-supervised
adaptation stage and a fully-supervised adaptation stage.
In the former, partial knowledge is transferred from multiple source domains to the target domain and fused therein.
Negative transfer between unmatching label spaces is mitigated via three new modules: domain attention, uncertainty
maximization and attention-guided adversarial alignment.
In the latter, knowledge is transferred in the unified label
space after a label completion process with pseudo-labels.
Extensive experiments on three different tasks - image classification, 2D semantic image segmentation, and joint 2D3D semantic segmentation - show that our method outperforms all competing methods significantly*
<!-- mDALU focuses on the multi-source Domain Adaptation and Label Unification problem (DALU), where datasets from multiple datasets from different domains as well as different set of labels are considered. In particular, the proposed framework uses a unified label space to define all semantic classes present, and a set of modules for domain adaptation and pseudo labeling are introduced to align different domains while generating pseudo labels w.r.t the unified label space for unlabeled samples in the source domain. The completed multi-source dataset with labels from the unified label space is used to train the final model. -->

## [xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jaritz_xMUDA_Cross-Modal_Unsupervised_Domain_Adaptation_for_3D_Semantic_Segmentation_CVPR_2020_paper.pdf)
*Venue: CVPR 2020*<br>
*Authors: Maximilian Jaritz, Tuan-Hung Vu, Raoul de Charette, Emilie Wirbel, Patrick Perez* <br>
*Abstract: Unsupervised Domain Adaptation (UDA) is crucial to tackle the lack of annotations in a new domain. There are many multi-modal datasets, but most UDA approaches are uni-modal. In this work, we explore how to learn from multi-modality and propose cross-modal UDA (xMUDA) where we assume the presence of 2D images and 3D point clouds for 3D semantic segmentation. This is challenging as the two input spaces are heterogeneous and
can be impacted differently by domain shift. In xMUDA, modalities learn from each other through mutual mimicking, disentangled from the segmentation objective, to prevent the stronger modality from adopting false predictions from the weaker one. We evaluate on new UDA scenarios including day-to-night, country-to-country and datasetto-dataset, leveraging recent autonomous driving datasets. xMUDA brings large improvements over uni-modal UDA
on all tested scenarios, and is complementary to state-ofthe-art UDA techniques. Code is available at https: //github.com/valeoai/xmuda.*
<!-- In xMUDA, Max et al. proposes a multi-modal framework to find a synergestic improvement on Unsupervised Domain Adaptation for the 2D camera image domain and the 3D LiDAR point cloud domain by exploiting the cross-modality of the two. Specifically, domain gaps on lighting conditions (day-to-night), environments (country-to-country), and sensor setup (dataset-to-dataset) are considered. The proposed learning scheme trains two unimodal branches, i.e. independent networks for image and LiDAR, by aligning features from both modalities in 3D space (image features are projected to corresponding 3D points) with symmetric KL divergence losses. Specifically, each stream outputs semantic labels for its input domain and for the cross-domain, and the cross-domain output and corresponding input domain are compared to match similarity. The cross modal learning, coupled with MinEnt and Deep LogCORAL for domain adaptation, yields significant improvements over the source-only baseline. -->

## [ConDA: Unsupervised Domain Adaptation for LiDAR Segmentation via Regualarized Domain Concatenation](https://arxiv.org/abs/2111.15242)
*Venue: ICRA 2023*<br>
*Authors: Lingdong Kong, Niamul Quader, Venice Erin Liong* <br>
*Abstract: Transferring knowledge learned from the labeled source domain to the raw target domain for unsupervised domain adaptation
(UDA) is essential to the scalable deployment of an autonomous driving system. State-of-the-art methods in UDA often employ a key idea:
utilizing joint supervision signals from both of the source domain (with
ground-truth) and target domain (with pseudo-labels) for self-training.
In this work, we improve and extend on this aspect. We present ConDA,
a Concatenation-based Domain Adaptation framework for LiDAR segmentation that: 1) constructs an intermediate domain consisting of finegrained interchange signals from both source and target domains without
destabilizing the semantic coherency of objects and background around
the ego-vehicle; and 2) utilizes the intermediate domain for self-training.
To improve both the network training on the source domain and selftraining on the intermediate domain, we propose an anti-aliasing regularizer and an entropy aggregator to reduce the negative effect caused
by the aliasing artifacts and noisy pseudo labels. Through extensive experiments, we demonstrate that ConDA significantly outperforms prior
arts in mitigating the domain gap not only in UDA, but also in other
DAs with minimum supervisions, such as semi-/weakly-supervised DAs.
Code will be publicly available3
.*

## [SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud](https://arxiv.org/abs/1809.08495)
*Venue: ICRA 2019*<br>
*Authors:Bichen Wu, Xuanyu Zhou, Sicheng Zhao, Xiangyu Yue, Kurt Keutzer* <br>
*Abstract: Earlier work demonstrates the promise of deeplearning-based approaches for point cloud segmentation; however, these approaches need to be improved to be practically
useful. To this end, we introduce a new model SqueezeSegV2
that is more robust to dropout noise in LiDAR point clouds.
With improved model structure, training loss, batch normalization and additional input channel, SqueezeSegV2 achieves
significant accuracy improvement when trained on real data.
Training models for point cloud segmentation requires large
amounts of labeled point-cloud data, which is expensive to
obtain. To sidestep the cost of collection and annotation,
simulators such as GTA-V can be used to create unlimited
amounts of labeled, synthetic data. However, due to domain
shift, models trained on synthetic data often do not generalize
well to the real world. We address this problem with a domainadaptation training pipeline consisting of three major components: 1) learned intensity rendering, 2) geodesic correlation
alignment, and 3) progressive domain calibration. When trained
on real data, our new model exhibits segmentation accuracy
improvements of 6.0-8.6% over the original SqueezeSeg. When
training our new model on synthetic data using the proposed
domain adaptation pipeline, we nearly double test accuracy on
real-world data, from 29.0% to 57.4%. Our source code and
synthetic dataset will be open-sourced.*

## [LidarNet: A Boundary-Aware Domain Adaptation Model for Point Cloud Semantic Segmentation](https://arxiv.org/abs/2003.01174)
*Venue: ArXiv*<br>
*Authors: Peng Jiang and Srikanth Saripalli* <br>
*Abstract: We present a boundary-aware domain adaptation model for LiDAR scan full-scene semantic segmentation (LiDARNet). Our model can extract both the domain private features and the domain shared features with a two-branch structure. We embedded Gated-SCNN into the segmentor component of LiDARNet to learn boundary information while learning to predict full-scene semantic segmentation labels. Moreover, we further reduce the domain gap by inducing the model to learn a mapping between two domains using the domain shared and private features. Additionally, we introduce a new dataset (SemanticUSL\footnote{The access address of SemanticUSL:\url{this https URL}}) for domain adaptation for LiDAR point cloud semantic segmentation. The dataset has the same data format and ontology as SemanticKITTI. We conducted experiments on real-world datasets SemanticKITTI, SemanticPOSS, and SemanticUSL, which have differences in channel distributions, reflectivity distributions, diversity of scenes, and sensors setup. Using our approach, we can get a single projection-based LiDAR full-scene semantic segmentation model working on both domains. Our model can keep almost the same performance on the source domain after adaptation and get an 8\%-22\% mIoU performance increase in the target domain.*

## [ePointDA: An End-to-End Simulation-to-Real Domain Adaptation Framework for LiDAR Point Cloud Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/16464/16271)
*Venue: AAAI-21*<br>
*Authors: Sicheng Zhao, Yezhen Wang23, Bo Li, Bichen Wu, Yang Gao, Pengfei Xu,Trevor Darrell, Kurt Keutzer* <br>
*Abstract: Due to its robust and precise distance measurements, LiDAR plays an important role in scene understanding for autonomous driving. Training deep neural networks (DNNs) on LiDAR data requires large-scale point-wise annotations, which are time-consuming and expensive to obtain. Instead, simulation-to-real domain adaptation (SRDA) trains a DNN using unlimited synthetic data with automatically generated labels and transfers the learned model to real scenarios. Existing SRDA methods for LiDAR point cloud segmentation mainly employ a multi-stage pipeline and focus on featurelevel alignment. They require prior knowledge of real-world statistics and ignore the pixel-level dropout noise gap and the spatial feature gap between different domains. In this paper, we propose a novel end-to-end framework, named ePointDA, to address the above issues. Specifically, ePointDA consists of three modules: self-supervised dropout noise rendering, statistics-invariant and spatially-adaptive feature alignment, and transferable segmentation learning. The joint optimization enables ePointDA to bridge the domain shift at the pixel-level by explicitly rendering dropout noise for synthetic LiDAR and at the feature-level by spatially aligning the features between different domains, without requiring the real-world statistics. Extensive experiments adapting from synthetic GTA-LiDAR to real KITTI and SemanticKITTI demonstrate the superiority of ePointDA for LiDAR point cloud segmentation.*
<!-- ePointDA focuses on Simluation to Real Domain Adaptation (SRDA) for LiDAR segmentation, with the motivation that annotated synthetic data is easier and efficient to generate with less limitations than that of annotating real life data. Specifically, ePointDA is an end-to-end framework with three modules: self-supervised dropout noise rendering, statistics-invariant and spatially-adaptive feature alignment, and transferable segmentation learning. The combination of three modules are what the authors propose to be necessary components for jointly reducing the domain shift from synthetic to real data, and experiment their framework with a modified SqueezeSegV2 network.  -->

## [Unsupervised Domain Adaptation in LiDAR Semantic Segmentation with Self-Supervision and Gated Adapters](https://arxiv.org/abs/2107.09783)
*Venue: ICRA 2022*<br>
*Authors: Mrigank Rochan, Shubhra Aich, Eduardo R. Corral-Soto, Amir Nabatchian, Bingbing Liu* <br>
*Abstract: In this paper, we focus on a less explored, but more realistic and complex problem of domain adaptation in LiDAR semantic segmentation. There is a significant drop in performance of an existing segmentation model when training (source domain) and testing (target domain) data originate from different LiDAR sensors. To overcome this shortcoming, we propose an unsupervised domain adaptation framework that leverages unlabeled target domain data for self-supervision, coupled with an unpaired mask transfer strategy to mitigate the impact of domain shifts. Furthermore, we introduce the gated adapter module with a small number of parameters into the network to account for target domain-specific information. Experiments adapting from both real-to-real and synthetic-to-real LiDAR semantic segmentation benchmarks demonstrate the significant improvement over prior arts.*

## [Unsupervised Domain Adaptation for Point Cloud Semantic Segmentation via Graph Matching](https://arxiv.org/pdf/2208.04510.pdf)
*Venue: IROS 2022*<br>
*Authors: Yikai Bian, Le Hui, Jianjun Qian, Jin Xie* <br>
*Abstract: Unsupervised domain adaptation for point cloud semantic segmentation has attracted great attention due to its effectiveness in learning with unlabeled data. Most of existing methods use global-level feature alignment to transfer the knowledge from the source domain to the target domain, which may cause the semantic ambiguity of the feature space. In this paper, we propose a graph-based framework to explore the local-level feature alignment between the two domains, which can reserve semantic discrimination during adaptation. Specifically, in order to extract local-level features, we first dynamically construct local feature graphs on both domains and build a memory bank with the graphs from the source domain. In particular, we use optimal transport to generate the graph matching pairs. Then, based on the assignment matrix, we can align the feature distributions between the two domains with the graph-based local feature loss. Furthermore, we consider the correlation between the features of different categories and formulate a category-guided contrastive loss to guide the segmentation model to learn discriminative features on the target domain. Extensive experiments on different synthetic-toreal and real-to-real domain adaptation scenarios demonstrate that our method can achieve state-of-the-art performance. Our code is available at https://github.com/BianYikai/PointUDA.*

## [Fake it, Mix it, Segment it: Bridging the Domain Gap Between Lidar Sensors](https://arxiv.org/abs/2212.09517)
*Venue: ArXiv*<br>
*Authors: Frederik Hasecke, Pascal Colling, Anton Kummert* <br>
*Abstract: Segmentation of lidar data is a task that provides rich, point-wise information about the environment of robots or autonomous vehicles. Currently best performing neural networks for lidar segmentation are fine-tuned to specific datasets. Switching the lidar sensor without retraining on a big set of annotated data from the new sensor creates a domain shift, which causes the network performance to drop drastically. In this work we propose a new method for lidar domain adaption, in which we use annotated panoptic lidar datasets and recreate the recorded scenes in the structure of a different lidar sensor. We narrow the domain gap to the target data by recreating panoptic data from one domain in another and mixing the generated data with parts of (pseudo) labeled target domain data. Our method improves the nuScenes to SemanticKITTI unsupervised domain adaptation performance by 15.2 mean Intersection over Union points (mIoU) and by 48.3 mIoU in our semi-supervised approach. We demonstrate a similar improvement for the SemanticKITTI to nuScenes domain adaptation by 21.8 mIoU and 51.5 mIoU, respectively. We compare our method with two state of the art approaches for semantic lidar segmentation domain adaptation with a significant improvement for unsupervised and semi-supervised domain adaptation. Furthermore we successfully apply our proposed method to two entirely unlabeled datasets of two state of the art lidar sensors Velodyne Alpha Prime and InnovizTwo, and train well performing semantic segmentation networks for both.*


## [PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation](https://openreview.net/pdf?id=ryxM3NrxIr)
*Venue: NeurIPS 2019*<br>
*Authors: Can Qin, Haoxuan You, Lichen Wang, C.-C. Jay Kuo, Yun Fu* <br>
*Abstract: Domain Adaptation (DA) approaches achieved significant improvements in a wide range of machine learning and computer vision tasks (i.e., classification, detection, and segmentation). However, as far as we are aware, there are few methods yet to achieve domain adaptation directly on 3D point cloud data. The unique challenge of point cloud data lies in its abundant spatial geometric information, and the semantics of the whole object is contributed by including regional geometric structures. Specifically, most general-purpose DA methods that struggle for global feature alignment and ignore local geometric information are not suitable for 3D domain alignment. In this paper, we propose a novel 3D Domain Adaptation Network for point cloud data (PointDAN). PointDAN jointly aligns the global and local features in multi-level. For local alignment, we propose Self-Adaptive (SA) node module with an adjusted receptive field to model the discriminative local structures for aligning domains. To represent hierarchically scaled features, node-attention module is further introduced to weight the relationship of SA nodes across objects and domains. For global alignment, an adversarial-training strategy is employed to learn and align global features across domains. Since there is no common evaluation benchmark for 3D point cloud DA scenario, we build a general benchmark (i.e., PointDA-10) extracted from three popular 3D object/scene datasets (i.e., ModelNet, ShapeNet and ScanNet) for cross-domain 3D objects classification fashion. Extensive experiments on PointDA-10 illustrate the superiority of our model over the state-of-the-art general-purpose DA methods.*

## [Self-Supervised Learning for Domain Adaptation on Point Clouds](https://arxiv.org/abs/2003.12641)
*Venue: WACV 2021*<br>
*Authors: Idan Achituve, Haggai Maron, GAl Chechik* <br>
*Abstract: Self-supervised learning (SSL) is a technique for learning useful representations from unlabeled data. It has been applied effectively to domain adaptation (DA) on images and videos. It is still unknown if and how it can be leveraged for domain adaptation in 3D perception problems. Here we describe the first study of SSL for DA on point clouds. We introduce a new family of pretext tasks, Deformation Reconstruction, inspired by the deformations encountered in sim-to-real transformations. In addition, we propose a novel training procedure for labeled point cloud data motivated by the MixUp method called Point cloud Mixup (PCM). Evaluations on domain adaptations datasets for classification and segmentation, demonstrate a large improvement over existing and baseline methods.*

<!-- While SSL for DA has been applied effectively to the image and video perception, it has not been explore as much in the 3D vision domain. As an early explorative work, the proposed method utilizes a set of pretext tasks called *Deformation Reconstruction*, augmented with Point Cloud Mixup(PCM) training procedure for domain adaptation on classification and segmentation of point clouds. -->

<!-- ## Unsupervised Domain Adaptation for LiDAR Panoptic Segmentation -->

---
<!-- <p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p> -->
<!-- Remove above link if you don't want to attibute -->