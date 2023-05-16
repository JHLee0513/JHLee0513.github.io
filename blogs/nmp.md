---
layout: post
---

# Deep Learning Based Methods For Autonomous Driving: A Survey

This is a relatively comprehensive survey of Deep Learning methods for different aspects of auotnomous Driving, including but not limited to perception for SDVs, end-to-end autonomous driving, and BEV perception. As one of the largest fields in Robotics and AI research, autonomous driving with DNNs have been explored with various methods, and I try to cover relatively significant, frequently cited, or state-of-the-art works from academia in this post. Please note that this is an on-going list as I will continue to update it, and furthermore is most definitely not an exhaustive list.

*disclaimer: various parts of this survey has been written with the help of ChatGPT, with proof-reading and further edits made by me.*

# Neural Motion Planning for Autonomous Driving
Learning based methods for modularized perception and motion planning. Neural Motion Planning methods typically conduct prediction and prediction coupled with trajectory sampling and cost volume generation to perform interpretable end-to-end motion planning.

## [End-to-end Interpretable Neural Motion Planner](https://arxiv.org/pdf/2101.06679.pdf)

![alt text](/images/SDV.pdf "SDV frameworks.")
<img src="/images/SDV.pdf" width="100px" height="50px" title="SDV Framework Comparison."/>

For an introduction to neural motion planning, let us consider *End-to-end Interpretable Neural Motion Planner*. As shown in the above diagram, traditional self driving stack follows a modular framework with the task of driving broken down into a set of sub-tasks, which are then engineered individually. However, tackling each sub-task in complete isolation is not ideal and can lead to sub-optimal performance. Possible deployment of each sub-system on separate hardware may also involve latency issues. Meanwhile, end to end driving has been previously explored with a black-box Deep Neural Network (DNN) trained to directly output control signals given sensory input. While showing the potential of learning a successful AI driver, such a method has been found difficult to interpret, and often fail to show successful generalization capabilities. Therefore, the proposed NMP focuses on maintaining a level of interpretability of the traditional SDV pipeline while allowing the entire framework to be trained end-to-end as a Deep Neural Network for efficiency and safety.

As shown in figure below from the paper, the End-to-end Interpretable Neural Motion Planner, or NMP, performs perception and prediction, costmap generation, trajectory sampling, and planning in a sequential flow. Let's break down each step:

![alt text](/images/NMP.png "End-to-end Interpretable Neural Motion Planner.")
<img src="/images/NMP.png" width="100px" height="50px" title="End-to-end Interpretable Neural Motion Planner."/>

**Perception:** The first step involves processing sensor input data, such as camera and lidar data, to extract relevant features for motion planning. The authors use a convolutional neural network (CNN) to extract these features, which are then used to generate a costmap. The authors use LiDAR projected in BEV frame (3D grid input mapped as an 'image') to process with a CNN for producing a BEV feature map. The encoder is also connected to a perception header that performs 3D object detection of vehicles from frame 0(present) to T-1 frames in the future, hence performing detection and forecasting.

**Costmap generation:** The costmap is a grid-based representation of the environment, where each cell represents the cost of traversing that region of the environment. The costmap is generated based on the features extracted in the perception step and is used to guide trajectory planning. The cost volume, joined with trajectory sampler, is trained with the Maximum Margin Planning loss given various trajectory demonstrations from real-world driving data.

**Trajectory sampling:** In this step, the authors generate a set of candidate trajectories based on the current vehicle state and the costmap. The strucuted minimization problem requires finding the trajectory with lowest cost as defined by the cost volume. For efficiency the authors use a bicycle model with Clothoid curves (also known as Euler spiral or Cornu spiral) to represent a set of possible 2D paths by the vehicle. Combining the sampled curved and velocity profiles, the authors are able to generate a set of plausible trajectories which are then evaluated on the predicted cost volume for Maximum Margin Planning loss. During training, the generated trajectories are treated as negative samples and the demonstrated trajectory as the ground truth for loss calculation.

**Trajectory planning:** Finally, during inference the predicted cost volume is used to evaluate sampled trajectories to select the best scoring trajectory for motion planning.

The NMP method proposed in this paper achieves state-of-the-art performance on a benchmark dataset and is shown to be interpretable, allowing the user to understand how the system arrived at its decisions. This level of interpretability is important for applications such as autonomous driving, where safety is of utmost importance. By combining the benefits of the traditional modular approach and the black-box end-to-end approach, the proposed NMP method represents a promising step forward in the field of interpretable autonomous driving.

*If the reader is interested in the joint perception and prediction task, I also recommend [PnPNet: End-to-End Perception and Prediction with Tracking in the Loop](https://arxiv.org/pdf/2005.14711.pdf), which was from published from the same lab.*

## [Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations](https://arxiv.org/pdf/2008.05930.pdf)

![alt text](/images/P3.png "Perceive, Predict, and Plan.")
<img src="/images/P3.png" width="100px" height="50px" title="Perceive, Predict, and Plan."/>

"Perceive, Predict, and Plan," by Sadat et al., introduces a novel end-to-end learnable neural motion planning framework. Its key difference to NMP is that its planning costs are consistent with the perception task, as they utilize the intermediate semantic representation as part of its cost volume generationn process. Instead of the feature map from the convolutional backbone processed by a perception head and cost volume head separately, their Perceive, Predict, and Plan, or P3, framework outputs a *semantic occupancy* representation that efficiently captures a dense semantic understanding of the surrounding environment, while feeding to same representation as cost function for max-margin loss based motion planner. xxx

The semantic activations produced by the proposed model are highly interpretable. The authors generate occupancy layers for each class of vehicles, bicyclists, and pedestrians, as well as occlusion layers that predict occluded objects. Moreover, by using the planned route of the self-driving vehicle, they can semantically differentiate vehicles by their interaction with the intended route, such as oncoming traffic versus crossing, which adds to the interpretability of the perception outputs. This differentiation can also potentially help the planner learn different subcosts for each category, such as different safety buffers for parked vehicles versus oncoming traffic.

![alt text](/images/P3_motionplanning.png "Motiong Planner costs in Perceive, Predict, and Plan.")
<img src="/images/P3_motionplanning.png" width="100px" height="50px" title="Perceive, Predict, and Plan."/>

The proposed model's sample-based learnable motion planner then takes these occupancy predictions and evaluates the associated risk of different maneuvers to find a safe and comfortable trajectory for the self-driving vehicle. This is done through an interpretable cost function used to cost motion-plan samples, which efficiently exploits the occupancy information. The model is trained end-to-end to imitate human driving while avoiding collisions and traffic infractions, and Fig. 1 provides an overview of the proposed approach. Overall, the authors show that their proposed model significantly outperforms state-of-the-art planners in imitating human behaviors while producing much safer trajectories.

Their experiments on large-scale real driving dataset and closed-loop simluations show that the proposed method significantly outperforms state-of-the-art planners in imitating human behaviors while producing much safer trajectories.


merge below paragraphs to the above:
The network achieves this by using a novel differentiable semantic occupancy representation that is explicitly used as cost by the motion planning process. Furthermore, the proposed model is learned end-to-end from human demonstrations. The experiments on a large-scale manual-driving dataset and closed-loop simulation show that the proposed model 

The method proposed in "Perceive, Predict, and Plan" is an end-to-end approach to self-driving that produces intermediate representations designed for safe planning and decision-making while maintaining interpretability. The authors make use of a map, the intended route, and raw LiDAR point-cloud to generate an intermediate semantic occupancy representation over space and time. These occupancy layers provide information about potential objects, including those with low probability, enabling perception of objects of arbitrary shape rather than just bounding boxes. This is a significant improvement over existing approaches that rely on object detectors that threshold activations and produce objects with only bounding box shapes, which can be problematic for safety.



## [MP3: A Unified Model to Map, Perceive, Predict and Plan](https://openaccess.thecvf.com/content/CVPR2021/papers/Casas_MP3_A_Unified_Model_To_Map_Perceive_Predict_and_Plan_CVPR_2021_paper.pdf)

![alt text](/images/MP3.png "MP3: A Unified Model to Map, Perceive, Predict and Plan.")
<img src="/images/MP3.png" width="100px" height="50px" title="MP3: A Unified Model to Map, Perceive, Predict and Plan."/>

The quest to achieve safe and efficient autonomous driving is an ongoing challenge that requires continuous advancements in perception, prediction, and planning capabilities. In recent years, Casas et al. has made significant strides towards this goal, proposing several end-to-end learnable models for self-driving vehicles. One of the key challenges in developing self-driving systems is the construction of high-definition maps (HD maps), which are expensive to create and maintain, and require high-precision localization systems. In their latest paper titled "MP3: A Unified Model to Map, Perceive, Predict and Plan," Casas et al. propose a mapless driving approach that can operate using raw sensor data and a high-level command. Their proposed model predicts intermediate representations in the form of an online map and the current and future state of dynamic agents, which are used by a novel neural motion planner to make interpretable decisions taking into account uncertainty. The MP3 approach is shown to be significantly safer, more comfortable, and better able to follow commands than baselines in challenging long-term closed-loop simulations and when compared to an expert driver in a large-scale real-world dataset.

## [Perceive, Attend, and Drive: Learning Spatial Attention for Safe Self-Driving](https://arxiv.org/pdf/2011.01153.pdf)

The paper proposes a method for autonomous driving called Sparse Attention Non-Markov Planning (SA-NMP) that utilizes sparse attention to improve computational efficiency and interpretability while maintaining high accuracy. SA-NMP consists of three components: sparse attention for feature extraction, Non-Markov prediction for multi-modal trajectory forecasting, and planning using a motion planner. The sparse attention mechanism is responsible for identifying relevant areas in the input data and passing them through the Non-Markov predictor, which outputs multi-modal trajectory predictions. Finally, the motion planner uses the predicted trajectories to generate a safe and efficient driving policy.

The novel contributions of this paper are twofold. Firstly, they introduce a sparse attention mechanism that allows the model to focus only on relevant areas of the input data, significantly reducing the computational cost and improving interpretability. Secondly, they propose the Non-Markov predictor, which can capture non-Markovian dependencies in the data by using a transformer-like architecture that models the sequential interactions between instances in the scene.

The experimental evaluation of the proposed method was conducted on two datasets: Drive4D and nuScenes v1.0. SA-NMP outperforms state-of-the-art methods in terms of planning L2, collision rate, and lane violation rate on both datasets. Additionally, the authors compare their learned attention mechanism to several baselines, demonstrating the superiority of their approach over static attention masks. The paper provides insights into the benefits of utilizing sparse attention mechanisms in the context of autonomous driving, which can improve the interpretability and computational efficiency of the model while maintaining high accuracy.

## [ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning](https://arxiv.org/pdf/2207.07601.pdf)

ST-P3 proposes a novel approach for autonomous driving, which integrates camera-based vision along with LiDAR-based pipelines to improve the performance of perception, prediction, and planning tasks. The paper's primary focus is on the design of an interpretable end-to-end framework, which takes raw sensor data as inputs and generates planning routes or control signals. By incorporating camera-based vision, the framework can better capture the environment's visual cues, providing a more complete picture of the surrounding scene. However, one of the primary challenges with vision-based methods is appropriately transforming feature representations from perspective views to the bird's eye view (BEV) space. To address this challenge, the paper proposes a novel approach for accumulating all past aligned features in 3D space before transforming to BEV, preserving geometry information at best and compensating for more robust feature representations of the current state. The paper also formulates the prediction task as future instance segmentation and incorporates an additional temporal model with a fusion unit to reason about the probabilistic nature of both past and future motions, resulting in a stronger version of scene representations. For motion planners, the paper constructs cost volumes to indicate the confidence of trajectories and indicates the most probable candidate with the help of a high-level command without HD map as guidance. Finally, the proposed framework achieves state-of-the-art performance on various benchmark datasets, demonstrating its potential for realistic application in autonomous driving.

## [Rules of the Road: Predicting Driving Behavior with a Convolutional Model of Semantic Interactions](https://arxiv.org/pdf/1906.08945.pdf)

The proposed method in the paper "Rules of the Road: Predicting Driving Behavior with a Convolutional Model of Semantic Interactions" comprises three main components: (1) a unique input representation of an entity and its surrounding world context, (2) a neural network that maps past and present world representation to future behaviors, and (3) one of several possible output representations of future behavior that can be integrated into a robot planning system. The model focuses on a single "target entity" and is designed to take all other world context, including other entities, into account, making it an entity-centric model. This approach allows for a deep understanding of the semantics of the static and dynamic environment, including traffic laws, driving conventions, and interactions between human and robot actors, which is crucial for self-driving robots operating in unconstrained urban environments.

## [Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer](https://arxiv.org/pdf/2207.14024.pdf)

The paper titled "Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer" addresses the challenges of developing safe and reliable autonomous vehicles, particularly in high-traffic-density scenes where a large number of obstacles and dynamic objects are involved in the decision making. The paper proposes a novel approach called Interpretable Sensor Fusion Transformer (InterFuser), which fuses information from multi-modal multi-view sensors and provides intermediate interpretable features as safety constraint heuristics to enhance driving safety. The authors experimentally validated the proposed method on several CARLA benchmarks with complex and adversarial urban scenarios, and their model outperformed all prior methods, ranking first on the public CARLA Leaderboard.

The safety enhanced controller of the autonomous vehicle is designed to ensure safe driving by utilizing a combination of interpretable features and waypoints generated by the transformer decoder. The low-level actions of the vehicle, namely lateral steering and longitudinal acceleration, are determined by a PID controller that aligns the vehicle to the desired heading and aims to reach the desired speed while taking into account the surrounding objects. The object density map is utilized to determine the existence of an object in a grid by either its existence probability in the grid or by identifying the local maximum in surrounding grids, and then predicting its future trajectory by propagating its historical dynamics with moving average. The maximum safe distance the vehicle can travel is then determined, and a linear programming problem is solved to derive the desired velocity with enhanced safety. Additionally, the predicted traffic rule is also used for safe driving, and the vehicle performs an emergency stop if the traffic light is not green or there is a stop sign ahead. While more advanced trajectory prediction methods and safety controllers can be used, the current controller is deemed sufficient for the task at hand, and future integration of these advanced algorithms is possible for more complex driving tasks.

## [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/pdf/1812.03079.pdf)

## [Imitation Is Not Enough: Robustifying Imitation with Reinforcement Learning for Challenging Driving Scenarios](https://arxiv.org/pdf/2212.11419.pdf)

# Vision to Control
Learning methods to directly map control/plan from visual input(RGB images, LiDAR, etc.)

## [Model-Based Imitation Learning for Urban Driving (MILE)](https://arxiv.org/pdf/2210.07729.pdf)

I believe MILE's approach is best explained by the people from Wayve that created the system, as I quote([link](https://wayve.ai/thinking/learning-a-world-model-and-a-driving-policy/)): "MILE’s main components are:

Observation encoder. Since autonomous driving is a geometric problem where it is necessary to reason in 3D about the static environment and dynamic agents, we first lift the image features to 3D. The 3D feature voxels are then sum-pooled to Bird’s-Eye View (BEV) using a predefined grid. Even after sum-pooling the voxel features to BEV, the high dimensionality of the BEV features is prohibitive for a probabilistic world model. Therefore, using a convolutional backbone, we further compress the BEV features to a one-dimensional vector.
Probabilistic modelling. The world model is trained to match the distribution of the prior distribution (a guess of what will happen after the executed action) to the posterior distribution (the evidence of what actually happened).
Decoders. The observation decoder and the BEV decoder have an architecture similar to StyleGAN [Karras et al. 2019]. The prediction starts as a learned constant tensor, and is progressively upsampled to the final resolution. At each resolution, the latent state is injected into the network with adaptive instance normalisation. This allows the latent states to modulate the predictions at different resolutions. Additionally, the driving policy outputs the vehicle control.
Temporal modelling. Time is modelled by a recurrent network that models the latent dynamics, predicting the next latent state from the previous latent state.
Imagination. From this observed past context, the model can imagine future latent states and use them to plan and predict actions using the driving policy. Future states can also be visualised and interpreted through the decoders.
From past observations, our model can imagine plausible diverse futures and plan different actions based on the predicted future. We demonstrate this with an example of MILE approaching an intersection. The traffic light is green, and we are following a vehicle."

![alt text](/images/MILE.png "Model-Baased Imitation Learning for Urban Driving.")
<img src="/images/MILE.png" width="80" height="40px" title="Model-Baased Imitation Learning for Urban Driving."/>

## [FIERY](https://arxiv.org/pdf/2104.10490.pdf)

## [End-to-end Driving via Conditional Imitation Learning](https://vladlen.info/papers/conditional-imitation.pdf)

## [VISTA 2.0: An Open, Data-driven Simulator for Multimodal Sensing and Policy Learning for Autonomous Vehicles](https://arxiv.org/abs/2111.12083)

# Self-supervision driven representation for driving

It needs to be noted that to reduce input complexity and improve generalization, aforementioned methods in developing planning networks for autonomous driving involves using intermediate representation that are derived from perception systems. These perception systems are developed via supervised learning with tasks such as segmentation and detection, and thus requires data labeling, and hence can become a major bottleneck towards scalable learning of driving.    

## CIL

## CIRL
<!-- 
# Perception & Prediction

## PnPNet

## LaneGCN

## VectorNet

## Detectron++

## TransFusion

## TnT

## TrajFormer

## WayFormer -->

# BEV Perception
I recommend [Delving into the Devils of Bird's-eye-view Perception: A Review, Evaluation and Recipe](https://arxiv.org/pdf/2209.05324.pdf).   

# Strucutured, Engineering Approaches

I highlight miscellaneous interesting work here

## [NVIDIA Safety Force Field](https://www.nvidia.com/content/dam/en-zz/Solutions/self-driving-cars/safety-force-field/an-introduction-to-the-safety-force-field-v2.pdf)

***"The Safety Force Field is built on a simple core concept:
Actors in traffic should apply a safety procedure or equivalent action before it is too late"***



# Miscellaneous Links
1. https://www.youtube.com/watch?v=PyOQibtWHI0
2. 

