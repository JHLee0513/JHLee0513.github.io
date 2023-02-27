---
layout: post
---

# Neural Motion Planning for Autonomous Driving

This is a survey of using Deep Neural Networks (DNNs) in the context of end-to-end learning for autonomous driving. As one of the hottest fields in Robotics and AI research, end-to-end driving with DNNs can be viewed with various methods, with overall relations to the decoupled task of vision and planning.

## [End-to-end Interpretable Neural Motion Planner](https://arxiv.org/pdf/2101.06679.pdf)

![alt text](/images/SDV.pdf "SDV frameworks.")
<img src="/images/SDV.pdf" width="100px" height="50px" title="SDV Framework Comparison."/>

A good paper to start of the discussion would be *End-to-end Interpretable Neural Motion Planner*. As shown in the above diagram, traditional self driving stack follows a modular framework with the task of driving broken into set of sub-tasks and engineered individually. However, engineering each sub-task in complete isolation is not ideal and can lead to sub-optimal performance. Meanwhile, possible deployment of each sub-system on separate hardware may also involve latency issues. On the other hand, end to end driving has been previously explored with a black-box Deep Neural Network (DNN) trained to directly output control signals given sensory input. While showing potential of learning a successful driver end-to-end, such a method has been found to be difficult to interpretable, and fail to show successful generalization capabilities. Therefore, the method proposed NMP method focuses on maintaining a level of interpretability while allowing the entire framework to be trained end-to-end as a Deep Neural Network.

The prposed NMP performs perception, costmap generation, trajectory sampling and planning in a sequential flow.

If the reader is interested in the joint perception and prediction task, I also recommend the paper [PnPNet: End-to-End Perception and Prediction with Tracking in the Loop](https://arxiv.org/pdf/2005.14711.pdf), which was from published by the same lab.

## [Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations](https://arxiv.org/pdf/2008.05930.pdf)

## [MP3: A Unified Model to Map, Perceive, Predict and Plan](https://openaccess.thecvf.com/content/CVPR2021/papers/Casas_MP3_A_Unified_Model_To_Map_Perceive_Predict_and_Plan_CVPR_2021_paper.pdf)

## [ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning](https://arxiv.org/pdf/2207.07601.pdf)

## [Rules of the Road: Predicting Driving Behavior with a Convolutional Model of Semantic Interactions](https://arxiv.org/pdf/1906.08945.pdf)

## [Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer](https://arxiv.org/pdf/2207.14024.pdf)

## [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/pdf/1812.03079.pdf)

# Self-supervision driven representation for driving

It needs to be noted that to reduce input complexity and improve generalization, aforementioned methods in developing planning networks for autonomous driving involves using intermediate representation that are derived from perception systems. These perception systems are developed via supervised learning with tasks such as segmentation and detection, and thus requires data labeling, and hence can become a major bottleneck towards scalable learning of driving.    

## CIL

## CIL++

# Perception & Prediction

## PnPNet

## LaneGCN

## VectorNet

## Detectron++

## TrajFormer

## TransFusion

# Miscellaneous Links
1. https://www.youtube.com/watch?v=PyOQibtWHI0
2. 

