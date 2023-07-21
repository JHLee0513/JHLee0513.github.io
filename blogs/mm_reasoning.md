---
layout: post
---

# Generalist Multimodal Reasoning (and Foundation Models)

This blog covers generalist multimodal reasoning models, namely Gato and PaLM-E. It also briefly covers relevant foundation models such as ViNT.

*Last update date: 2023-07-20*

## [A Generalist Agent](https://openreview.net/forum?id=1ikK0kHjvj)

*Venue: Transactions on Machine Learning Research (TMLR)*<br>  
*Authors: Li Yi, Boqing Gong, Thomas Funkhouser* <br>

Gato was the first paper I learned about generalist models, first hearing of its introduction from [Lex Fridman's podcast](https://www.youtube.com/watch?v=aGBLRlLe7X8). As Oriol Vinyals explains, large sequence models from [Gopher](https://arxiv.org/abs/2112.11446), [Chinchilla](https://arxiv.org/abs/2203.15556), vision-language model [Flamingo](https://arxiv.org/abs/2204.14198), Gato is a vision-language-action model that not only processes visual and text but also handles action-based output.

Gato is based on the transformers network, with relevant architectures such as Decision Transformers and Trajectory Transformer. Notably, Gato scales up to 1.8B parameters, showing transfer of skills with a single set of model weights able to perform various robotic, textual, visual and multi-modal tasks. In order to process multi-modality, Gato accepts inupt such as text, image, and continous/discrete actions to then be tokenized into an input sequence. Each input modality goes through an engineered pre-processing step. The model is trained in an auto-regressive manner with expert data, and therefore can be seen as a variant of supervised learning (or Behavior Cloning). In order to handle multi-task learning, Gato is provided with an input prompt designed to provie task/domain context, which arguably reduces the impact of catastrophic forgetting. What the researchers ended up with was , and thruogh experiments show this emergent capabilities grow with the model scale.

However, as Oriol has pointed out in his [tweet](https://twitter.com/OriolVinyalsML/status/1529892826099724306), Gato is certainly the beginning, as soon we are introduced PaLM-E.

## [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io/assets/palm-e.pdf)
*Venue: ICRA 2022*<br>
*Authors: Mrigank Rochan, Shubhra Aich, Eduardo R. Corral-Soto, Amir Nabatchian, Bingbing Liu* <br>

With the model scale grown as large as up to 540B parameters, researchers were able to realize greater synergistic transfer between tasks, where multi-task trained model performs much better than separately trained mdoels. This is notably a surprised compared to Gato, where it would still perform worse than its teacher (models specifically tuned to one task) for some cases. (It's still to be noted that this synergistic growth was something authors of Gato were expecting may occur with model scale larger than that of Gato).
<!-- 
More recently, the idea has extended to robotic navigation, and I breifly explain the work of ViNT in this post as well.

## [ViNT: A Foundation Model for Visual Navigation](https://arxiv.org/abs/2306.14846)
*Venue: arXiv*<br>
*Authors: Dhruv Shah†, Ajay Sridhar†, Nitish Dashora†,
Kyle Stachowicz, Kevin Black, Noriaki Hirose, Sergey Levine* <br>
 -->

