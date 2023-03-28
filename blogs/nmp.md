---
layout: post
---

# Deep Learning Based Methods For Autonomous Driving: A Survey

This is a relatively comprehensive survey of Deep Learning methods for different aspects of auotnomous Driving, including but not limited to perception for SDVs, end-to-end autonomous driving, and BEV perception. As one of the largest fields in Robotics and AI research, autonomous driving with DNNs have been explored with various methods, and I try to cover relatively significant, frequently cited, or state-of-the-art works from academia in this post. Please note that this is an on-going list as I will continue to update it, and furthermore is most definitely not an exhaustive list.

# Neural Motion Planning for Autonomous Driving


## [End-to-end Interpretable Neural Motion Planner](https://arxiv.org/pdf/2101.06679.pdf)

![alt text](/images/SDV.pdf "SDV frameworks.")
<img src="/images/SDV.pdf" width="100px" height="50px" title="SDV Framework Comparison."/>

For an introduction to neural motion planning, let us consider *End-to-end Interpretable Neural Motion Planner*. As shown in the above diagram, traditional self driving stack follows a modular framework with the task of driving broken down into a set of sub-tasks, which are then engineered individually. However, tackling each sub-task in complete isolation is not ideal and can lead to sub-optimal performance. Possible deployment of each sub-system on separate hardware may also involve latency issues. Meanwhile, end to end driving has been previously explored with a black-box Deep Neural Network (DNN) trained to directly output control signals given sensory input. While showing the potential of learning a successful AI driver, such a method has been found difficult to interpret, and often fail to show successful generalization capabilities. Therefore, the proposed NMP focuses on maintaining a level of interpretability of the traditional SDV pipeline while allowing the entire framework to be trained end-to-end as a Deep Neural Network for efficiency and safety.

As shown in figure below from the paper, the End-to-end Interpretable Neural Motion Planner, or NMP, performs perception and prediction, costmap generation, trajectory sampling, and planning in a sequential flow. Let's break down each step:

![alt text](/images/NMP.png "End-to-end Interpretable Neural Motion Planner.")
<img src="/images/NMP.png" width="100px" height="50px" title="End-to-end Interpretable Neural Motion Planner."/>

**Perception:** The first step involves processing sensor input data, such as camera and lidar data, to extract relevant features for motion planning. The authors use a convolutional neural network (CNN) to extract these features, which are then used to generate a costmap. The authors use LiDAR projected in BEV frame (3D grid input mapped as an 'image') to process with a CNN for producing a BEV feature map. The encoder is also connected to a perception header that performs 3D object detection of vehicles from frame 0(present) to T-1 frames in the future, hence performing detection and forecasting.

**Costmap generation:** The costmap is a grid-based representation of the environment, where each cell represents the cost of traversing that region of the environment. The costmap is generated based on the features extracted in the perception step and is used to guide trajectory planning. The cost volume, joined with trajectory sampler, is trained with the Maximum Margin Planning loss given various trajectory demonstrations from real-world driving data.

**Trajectory sampling:** In this step, the authors generate a set of candidate trajectories based on the current vehicle state and the costmap. They use a probabilistic roadmap (PRM) algorithm to generate these trajectories, which ensures that the trajectories are collision-free.???

**Trajectory planning:** Finally, the authors use a recurrent neural network (RNN) to select the best trajectory from the set of candidate trajectories generated in the previous step. The RNN takes as input the current vehicle state, the costmap, and the candidate trajectories, and outputs the optimal trajectory.

The NMP method proposed in this paper achieves state-of-the-art performance on a benchmark dataset and is shown to be interpretable, allowing the user to understand how the system arrived at its decisions. This level of interpretability is important for applications such as autonomous driving, where safety is of utmost importance. By combining the benefits of the traditional modular approach and the black-box end-to-end approach, the proposed NMP method represents a promising step forward in the field of interpretable autonomous driving.

If the reader is interested in the joint perception and prediction task, I also recommend the paper [PnPNet: End-to-End Perception and Prediction with Tracking in the Loop](https://arxiv.org/pdf/2005.14711.pdf), which was from published by the same lab.


## [Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations](https://arxiv.org/pdf/2008.05930.pdf)

Sadat et al. proposed a paper titled "Perceive, Predict, and Plan," which introduces a novel end-to-end learnable network for self-driving vehicles. Unlike the previous neural motion planners, this network performs joint perception, prediction, and motion planning, and produces interpretable intermediate representations. One limitation of the previous work, the NMP, was that it only handled object-level representations. In contrast, the proposed network uses a dense semantic representation as the intermediate representation for autonomous driving. The network achieves this by using a novel differentiable semantic occupancy representation that is explicitly used as cost by the motion planning process. Furthermore, the proposed model is learned end-to-end from human demonstrations. The experiments on a large-scale manual-driving dataset and closed-loop simulation show that the proposed model significantly outperforms state-of-the-art planners in imitating human behaviors while producing much safer trajectories.

The method proposed in "Perceive, Predict, and Plan" is an end-to-end approach to self-driving that produces intermediate representations designed for safe planning and decision-making while maintaining interpretability. The authors make use of a map, the intended route, and raw LiDAR point-cloud to generate an intermediate semantic occupancy representation over space and time. These occupancy layers provide information about potential objects, including those with low probability, enabling perception of objects of arbitrary shape rather than just bounding boxes. This is a significant improvement over existing approaches that rely on object detectors that threshold activations and produce objects with only bounding box shapes, which can be problematic for safety.

The semantic activations produced by the proposed model are highly interpretable. The authors generate occupancy layers for each class of vehicles, bicyclists, and pedestrians, as well as occlusion layers that predict occluded objects. Moreover, by using the planned route of the self-driving vehicle, they can semantically differentiate vehicles by their interaction with the intended route, such as oncoming traffic versus crossing, which adds to the interpretability of the perception outputs. This differentiation can also potentially help the planner learn different subcosts for each category, such as different safety buffers for parked vehicles versus oncoming traffic.

The proposed model's sample-based learnable motion planner then takes these occupancy predictions and evaluates the associated risk of different maneuvers to find a safe and comfortable trajectory for the self-driving vehicle. This is done through an interpretable cost function used to cost motion-plan samples, which efficiently exploits the occupancy information. The model is trained end-to-end to imitate human driving while avoiding collisions and traffic infractions, and Fig. 1 provides an overview of the proposed approach. Overall, the authors show that their proposed model significantly outperforms state-of-the-art planners in imitating human behaviors while producing much safer trajectories.

## [MP3: A Unified Model to Map, Perceive, Predict and Plan](https://openaccess.thecvf.com/content/CVPR2021/papers/Casas_MP3_A_Unified_Model_To_Map_Perceive_Predict_and_Plan_CVPR_2021_paper.pdf)

The quest to achieve safe and efficient autonomous driving is an ongoing challenge that requires continuous advancements in perception, prediction, and planning capabilities. In recent years, the research group led by Casas et al. has made significant strides towards this goal, proposing several end-to-end learnable models for self-driving vehicles. One of the key challenges in developing self-driving systems is the construction of high-definition maps (HD maps), which are expensive to create and maintain, and require high-precision localization systems. In their latest paper titled "MP3: A Unified Model to Map, Perceive, Predict and Plan," Casas et al. propose a mapless driving approach that can operate using raw sensor data and a high-level command. Their proposed model predicts intermediate representations in the form of an online map and the current and future state of dynamic agents, which are used by a novel neural motion planner to make interpretable decisions taking into account uncertainty. The MP3 approach is shown to be significantly safer, more comfortable, and better able to follow commands than baselines in challenging long-term closed-loop simulations and when compared to an expert driver in a large-scale real-world dataset.

## [ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning](https://arxiv.org/pdf/2207.07601.pdf)

ST-P3 proposes a novel approach for autonomous driving, which integrates camera-based vision along with LiDAR-based pipelines to improve the performance of perception, prediction, and planning tasks. The paper's primary focus is on the design of an interpretable end-to-end framework, which takes raw sensor data as inputs and generates planning routes or control signals. By incorporating camera-based vision, the framework can better capture the environment's visual cues, providing a more complete picture of the surrounding scene. However, one of the primary challenges with vision-based methods is appropriately transforming feature representations from perspective views to the bird's eye view (BEV) space. To address this challenge, the paper proposes a novel approach for accumulating all past aligned features in 3D space before transforming to BEV, preserving geometry information at best and compensating for more robust feature representations of the current state. The paper also formulates the prediction task as future instance segmentation and incorporates an additional temporal model with a fusion unit to reason about the probabilistic nature of both past and future motions, resulting in a stronger version of scene representations. For motion planners, the paper constructs cost volumes to indicate the confidence of trajectories and indicates the most probable candidate with the help of a high-level command without HD map as guidance. Finally, the proposed framework achieves state-of-the-art performance on various benchmark datasets, demonstrating its potential for realistic application in autonomous driving.

## [Rules of the Road: Predicting Driving Behavior with a Convolutional Model of Semantic Interactions](https://arxiv.org/pdf/1906.08945.pdf)

## [Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer](https://arxiv.org/pdf/2207.14024.pdf)

## [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/pdf/1812.03079.pdf)

# Vision to Control
Learning methods to directly map control/plan from visual input(RGB images, LiDAR, etc.)

## [CIRL]()

## [VISTA]()

## [VISTA 2.0]()

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

# BEV Perception
I recommend [Delving into the Devils of Bird's-eye-view Perception: A Review, Evaluation and Recipe](https://arxiv.org/pdf/2209.05324.pdf).   

# Miscellaneous Links
1. https://www.youtube.com/watch?v=PyOQibtWHI0
2. 

