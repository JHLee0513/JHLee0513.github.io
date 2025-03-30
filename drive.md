[comment]: <> (## Portfolio)

[comment]: <> (---)

## Learning to Drive: the story on Robotic Perception

There's been remarkable progress and massive attention on autonomous driving. While doing a literature survey on
approaching the problem, I have also been surveying available libraries from the industry. In this blog I go over
Nvidia's Drive system, showcasing it's system w.r.t other academic papers, and some demonstration on using it (TODO).
While this blog mainly focuses on the perception aspect, I hope to cover the learning and control, and more end-to-end
approaches in the future.


### Perception

[Paper](https://openreview.net/forum?id=AL4FPs84YdQ) [Website](https://sites.google.com/view/terrain-traversability/home)
<img src="images/warthog.png?raw=true"/>
<img src="images/canal.gif?raw=true"/>
<b>
<br>
5th Conference on Robot Learning (CoRL) 2021
</b>
<p>
Abstract: Producing dense and accurate traversability maps is crucial for autonomous off-road navigation. In this paper, we focus on the problem of classifying terrains into 4 cost classes (free, low-cost, medium-cost, obstacle) for traversability assessment. This requires a robot to reason about both semantics (what objects are present?) and geometric properties (where are the objects located?) of the environment. To achieve this goal, we develop a novel Bird's Eye View Network (BEVNet), a deep neural network that directly predicts a local map encoding terrain classes from sparse LiDAR inputs. BEVNet processes both geometric and semantic information in a temporally consistent fashion. More importantly, it uses learned prior and history to predict terrain classes in unseen space and into the future, allowing a robot to better appraise its situation. We quantitatively evaluate BEVNet on both on-road and off-road scenarios and show that it outperforms a variety of strong baselines.
</p>

## Research Projects
### Into the Wild: Robust Offroad Driving with Deep Perception

[comment]: <> (<a href='pdf/JoonHo_thesis.pdf'>PDF</a>)
<img src="images/canal.png?raw=true"/>
<img src="images/snow.gif?raw=true"/>
<img src="images/weeds_combined.gif?raw=true"/>
<br>
<b>Undergraduate Honors Thesis</b>
<br>

[PDF](/pdf/JoonHo_thesis.pdf)
<br>
Demo Videos: [Weeds](https://youtu.be/Ze9WJevj-Hw) [Snow](https://youtu.be/w5pjYyfmYsI)
<br>

<p>
Abstract: The task of autonomous offroad driving yields great potential for various beneficial applications, including but not limited to remote disaster relief, environment survey, and agricultural robotics. While achieving the task of robust offroad driving poses relatively new, interesting challenges to tackle, the most important requirement for a successful offroad autonomy is observed to be an effective understanding of the vehicle surrounding for robust navigation and driving. Therefore, in this thesis we tackle the task of scene understanding for autonomous offroad driving. We formulate the task of scene understanding as a traversability classification task, and develop a multimodal perception framework that extracts semantic knowledge. As our key contribution we propose a multimodal perception framework that uses convolutional neural networks with image and LiDAR input. The pipeline generates semantic knowledge from input data for robust mapping, planning, and control in the wild environment. We evaluate our method by integrating it into an autonomy stack and demonstrating its performance in a set of environments under various weather conditions.
</p>

---


### References:
1. Autonomous systems, a review.
2. 

---
<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
