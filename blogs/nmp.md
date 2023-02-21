---
layout: post
---

# Neural Motion Planning for Autonomous Driving

This is a survey of using Deep Neural Networks (DNNs) in the context of end-to-end learning for autonomous driving. As one of the hottest fields in Robotics and AI research, end-to-end driving with DNNs can be viewed with various methods, with overall relations to the decoupled task of vision and planning.

## End-to-end Interpretable Neural Motion Planner

A good paper to start of the discussion would be [*End-to-end Interpretable Neural Motion Planner*](link). 

![alt text](/images/SDV.pdf "SDV frameworks.")

<img src="/images/SDV.pdf" width="100px" height="50px" title="SDV Framework Comparison."/>

As shown above, the founding motivation for e2e-NMP is the interpretability. Traditional self driving stack follows the modular framework as illustrated above, where the complex task of driving is broken down into a set of sub-tasks. However, engineering each sub-task in complete isolation is unideal and can lead to sub-optimal performance. Meanwhile, possible deployment of each sub-system on separate hardware may also involve latency issues.

Therefore, the method proposed in e2e-NMP focuses on maintaining the similiar level of interpretability while allowing the entire framework to be trained end-to-end as a Deep Neural Network.

## Perceive, Predict, Plan

## MP3: A Unified Model to Map, Perceive, Predict and Plan

## Rules of the Road

## LaneGCN

## VectorNet

## ChauffeurNet

## Detectron++

## TrajFormer

## InterFuster

## TransFusion


