---
layout: post
---

# LLMs, General Multimodal Reasoning, and Foundation Models

*Last update date: 2023-10-24*

## Preliminaries

As part of preliminaries, or more of an introduction, I first summarize FiLM and Flamingo.

### [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)

*Date: Sept 2017*

* A simple yet effective way to guide extracted features toward certain goal i.e. guide extracted image features based on natural language instruction
* Showed substantial improvements when applied to CNN+GRU Network for the CLEVER dataset that involves tasks such as answering the shape of the object of interested from a cluttered environment.

### [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)

*Date: April 2022*

* Family of Vision Language Models with multi-modal capabilities such as VQA
* Supports video and image processing in a few-shot setting where it can perform various VQA such as explaning questions or captioning an iamage under ICL(In-Context Learning) settings
* Architecture-wise, they exploited large-scale learning by using frozen image encoder and LM, and only finetuning a small set of layers to finetune the connecting modules on a mix of datasets including ALIGN and M3W
* Model scaled up to 80B for performance gain via scale (3B, 7B, 80B)
* Uses different variants of LLMs such as Gopher(LLM trained at Deepmind) and Chinchilla(scaled up version of Gopher data-wise)

### CLIP

### TransporterNet

## Main Survey

## [A Generalist Agent](https://openreview.net/forum?id=1ikK0kHjvj)

*Date: Nov 2022*

* Discussed in [Lex Fridman's podcast](https://www.youtube.com/watch?v=aGBLRlLe7X8).
* Inspired from sequence models from [Gopher](https://arxiv.org/abs/2112.11446), [Chinchilla](https://arxiv.org/abs/2203.15556), vision-language model [Flamingo](https://arxiv.org/abs/2204.14198), Gato is a vision-language-action model that processes visual, text and action.
* Gato uses transformer architecture, and considers generalized sequence modeling problem, similar to the Decision Transformer.
* Related to Decision Transformer and TrajFormer, Gato works on treating large array of tasks as a sequential modeling problem i.e. everything from image captioning and question answering to playing games all represented by (state, action) trajectories.
* Gato scales up to 1.8B parameters, therefore not at <it> crazy scale </it>, but still very big nontheless. More importantly, they focus on training Gato on this extreme multi-task, multi-modal setting to observe relatively feasible zero shot transfer.
* Gato shows some transferability of skills with a single set of model weights able to perform various robotic, textual, visual and multi-modal tasks. 
* Gato accepts inupt such as text, image, and continous/discrete actions to then be tokenized into an input sequence. Each input modality goes through an engineered pre-processing step. The model is trained in an auto-regressive manner with expert data, and therefore can be seen as a variant of supervised learning (or Behavior Cloning).
* In order to handle multi-task learning, Gato is provided with an task-specific prompt alongside the actual intput to provide task/domain context, which arguably reduces the impact of catastrophic forgetting. Experiments show that better transferability capabilities grow with the model scale.

## [CLIPort: What and Where Pathways for Robotic Manipulation](https://cliport.github.io/)

*Date: Sept 2021*

* Released before Gato, focuses on robotic manipulation
* Imitation Learning Agent, in line with other generalist models such as Gato, BC-Z and RT-1.
* Fixed to tabletop object manipulation with top down viewpoint
* A two-stream architecture processes RGB for semantic (frozen CLIP encoder) and RGBD for geometric reasoning
* The two-stream output is fused at output level with TransporterNet as the overall meta architecture, where the method outputs what to move and where to move according to the TransporterNet framework
* Showed generalization to different objects, tasks, and instructions.

## [BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning](https://sites.google.com/view/bc-z/home)
*Date: Feb 2022*
* BC-Z aims to build large-scale data and apply Behavioral Cloning to build a conditional imitation learning model that is able to achieve zero-shot task transfer
* In terms of learning (i.e. losses), a relatively straightforward behavior cloning loss was used, where the learning policy outputs 6 DoF pose for end effector arm and a continous value for gripper angle.
* RGB image fed to ResNet-18 as policy
* Image as state, encoded video (showing example demonstration) or language instruction as command
* Dataset includes human demonstrations but also on-policy data based on HG-DAgger, where people corrected mistakes made by the policy.
* FiLM layers used to guide policy network based on instruction
* Video encoder network trained to output similar latent codes as the text encoder, which is a frozen Language Model e.g. CLIP text encoder
* Model shows some generalization to holdout tasks, but not as impressive when already knowing the performance of RT-1, RT-2, and RT-X.
* Overall BC-Z could handle 38% unseen tasks on easy setting, and 32% and 4% on the harder setting, where the latter is video conditioned. This shows that video feature extraction has a large room for improvement (HG-DAgger has also shown crucial for any of its successes)
* Demonstrated that frozen LLMs work well as encoders for compressing instructions into latent code for imitation learning

## [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)

*Date: Dec 2022*
*(RT-1 is also quite well explained in [this blog by Google](https://blog.research.google/2022/12/rt-1-robotics-transformer-for-real.html) )*

* Uses the transformer architecture for efficient large-scale learning
* Image as input state is first encoded with EfficientNetB3, with additional FiLM layers for goal conditioning features
* Improved feature compression with token learner (such that the transformer does not need to process all image patches)
* Final Transformer module processs the guided image feature tokens with natual language instruction as query to produce set of action sequences. The output action space includes poses for the arm and base, with a discrete token that functions as a switch to indicate which mode(arm/base) to operate.
* RT-1 showed significant improvements over Gato, BC-Z mainly due to its improved architecture and increased data scale. For instance, BC-Z 100 tasks with up to 30K demonstrations, while RT-1 has over 700 specified tasks with over 130K episodes collected on a fleet of 13 robots, a clear upgrade in scale. 

## [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io/assets/palm-e.pdf)

*Date: Mar 2023*

* A clear showcase of scale achieving even greater results
* Model as large as 540B parameters trained to achieve even stronger positive transfer across tasks (PaLM-E actively performs better than all specialized models while that wasn't always the case for Gato)
* Training the full model shows better performance than training only a subset of the model.
* Positive transfer is observed, where for each task the multi-task learning provides better performance per-task than training on the single task alone.
* Spiritual successor to Gato, and hence learned tasks relating to very general set of tasks such as playing gamess and VQA.
* Strong zero shot transfer capabilities, where the model is able to perform tasks without any further training.


## [VC-1: Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?](https://eai-vc.github.io/)

*Date: March 2023*

* Exploration of Learned Visual Representations for embodied AI tasks
* Authors observe that at the time of their experiments, the Masked AutoEncoder showed most promising results, and therefore further scale this method to train ViT-L and ViT-B on their large scale syntehtic embodieid AI benchmark termed CortexBench.
* (Paper considers evaluation where frozen weights of the image encoder is used to perform RL or IL)
* Larger scale dataset once against showed improved performances
* <b>However, the proposed method is not a clear winner on all tasks</b>, as the proposed MAE with CortexBench data performed best on average but not best overall.
* Authors show that finetuning leads to significant performance gains, while <b>visualizing the final attention layer also shows that pretrained models in general attend to the whole image while the finteuned model attends specifically to the regions of interest, such as the object being manipulated or teh end affector of the robot.</b>
* What I think: Based on the last finding, it seems the key is then being able to quickly attend to the right regions for given IL, RL task at hand.

## [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://robotics-transformer2.github.io/assets/rt2.pdf)

*Date: July 2023*
*(Also very well explained in their blog [here](https://www.deepmind.com/blog/rt-2-new-model-translates-vision-and-language-into-action))*


* RT-2 harnesses the VLM knowledge of PALM-E and PaLI-X with the robotic capabilities of RT-1 to show that general multi-modal (VL) reasoning and robotic control forms a positive transfer.
* Given input RGB image and natural language instructino, RT-2 predicts natural language output that usually includes a plan in addition to the specific control actions i.e. robot arm and base movement.
* Therefore, RT-2 provides exact movement plans instead of a set of actions that involved interaction by a low-level controller i.e. as explored in PaLM-E.
* RT-2 performance excels in the unseen setings in particular - unseen objects, backgrounds, environments and task.
* RT-2 specifically outputs a natural language response with action sequence denoted by 'Action:' key term to output pose sequence output for parsing into control.

## [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://robotics-transformer-x.github.io/)

*Date: October 2023*

Not so long after the impressive results from RT-2, a full-fledged large scale data movement has been proposed via the Open X Embodiment.

* Aggregated X robotic datasets to generate a large scale, diverse robotic manipulation dataset in light of building a foundational benchmark dataset similar to ImageNet was for Computer Vision.
* Models RT-1 and RT-2 trained on the new Open X dataset, dubbed RT-X-1 and RT-X-2, both outperform prior SOTA as well as themselves with the new dataset. 
* The Open X-Embodiment benchmark has data from 22 robots with 21 institutions as collaborators, on 527 skills (160266 tasks).
* As much as the scale is impressive, the benchmark data is released for open use.
* What I think: One follow-up question is how this may be distilled to more efficient models or how the data may be considered for more efficient, smaller-scale training.
