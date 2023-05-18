[comment]: <> (## Portfolio)

[comment]: <> (---)

## Publications 

### TerrainNet: Visual Modeling of Complex Terrain for High-speed, Off-road Navigation
[[Paper]](https://arxiv.org/abs/2303.15771) [[Website]](https://sites.google.com/view/visual-terrain-modeling)
<br>
*Xiangyun Meng, Nathan Hatch, Alexander Lambert, Anqi Li, Nolan Wagener, Matthew Schmittle, <b>JoonHo Lee</b>, Wentao Yuan, Zoey Chen, Samuel Deng, Greg Okopal, Dieter Fox, Byron Boots, Amirreza Shaban*
<br>
<img src="images/terrainnet.png?raw=true"/>

<!-- <b>
<br>
5th Conference on Robot Learning (CoRL) 2021
</b> -->
<p>
Effective use of camera-based vision systems is essential for robust performance in autonomous off-road driving, particularly in the high-speed regime. Despite success in structured, on-road settings, current end-to-end approaches for scene prediction have yet to be successfully adapted for complex outdoor terrain. To this end, we present TerrainNet, a vision-based terrain perception system for semantic and geometric terrain prediction for aggressive, off-road navigation. The approach relies on several key insights and practical considerations for achieving reliable terrain modeling. The network includes a multi-headed output representation to capture fine- and coarse-grained terrain features necessary for estimating traversability. Accurate depth estimation is achieved using self-supervised depth completion with multi-view RGB and stereo inputs. Requirements for real-time performance and fast inference speeds are met using efficient, learned image feature projections. Furthermore, the model is trained on a large-scale, real-world off-road dataset collected across a variety of diverse outdoor environments. We show how TerrainNet can also be used for costmap prediction and provide a detailed framework for integration into a planning module. We demonstrate the performance of TerrainNet through extensive comparison to current state-of-the-art baselines for camera-only scene prediction. Finally, we showcase the effectiveness of integrating TerrainNet within a complete autonomous-driving stack by conducting a real-world vehicle test in a challenging off-road scenario.
</p>

### Semantic Terrain Classification for Off-Road Autonomous Driving
[[Paper]](https://openreview.net/forum?id=AL4FPs84YdQ) [[Website]](https://sites.google.com/view/terrain-traversability/home)
<br>
*Amirreza Shaban, Xiangyun Meng, <b>JoonHo Lee</b>, Byron Boots, Dieter Fox*
<b>
<br>
5th Conference on Robot Learning (CoRL) 2021
</b>
<img src="images/warthog.png?raw=true"/>
<img src="images/canal.gif?raw=true"/>
<p>
Abstract: Producing dense and accurate traversability maps is crucial for autonomous off-road navigation. In this paper, we focus on the problem of classifying terrains into 4 cost classes (free, low-cost, medium-cost, obstacle) for traversability assessment. This requires a robot to reason about both semantics (what objects are present?) and geometric properties (where are the objects located?) of the environment. To achieve this goal, we develop a novel Bird's Eye View Network (BEVNet), a deep neural network that directly predicts a local map encoding terrain classes from sparse LiDAR inputs. BEVNet processes both geometric and semantic information in a temporally consistent fashion. More importantly, it uses learned prior and history to predict terrain classes in unseen space and into the future, allowing a robot to better appraise its situation. We quantitatively evaluate BEVNet on both on-road and off-road scenarios and show that it outperforms a variety of strong baselines.
</p>

## Research Projects

### RACER: Robotic Autonomy in Complex Environments with Resiliency
As part of the UW team, I am working on robotic autonomy for challenging offroad environments.
[Press release](https://www.darpa.mil/news-events/2022-01-13)
We recently released a few testing videos ran at DirtFish:
[Test 1](https://youtu.be/ibNW6Vezqpc)
[Test 2](https://www.youtube.com/watch?v=7-G9uPJ07uQ)

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

### Into the Wild: Robust Offroad Driving with Deep Perception
[comment]: <> (<a href='pdf/clipcap++_report.pdf'>PDF</a>)
<br>
<b>NLP capstone project (completed during MS)</b>
<br>
[PDF](/pdf/clipcap++_report.pdf)
<p>
Abstract: Modern research in Image Captioning typically utilizes transformers to achieve high accuracy. However, these methods at a large scale require both substantial amounts of data and compute, which makes training often challenging. To address this issue, we propose to train a mapping network between a pretrained image encoder and text decoder for efficiency. Our approach, based on ClipCap, explores improved utilization of the pretrained models, yielding improved performance on the COCO Captions dataset while training only the mapping network. This report has been developed as part of a Capstone class (CSE481N, University of Washington), and our code is available on [https://github.com/quocthai9120/UW-NLP-Capstone-SP22](https://github.com/quocthai9120/UW-NLP-Capstone-SP22).
</p>

---

## Miscellaneous Projects

### Dirt Segmentation
[Technical Report](/pdf/DL_report.pdf)
<br>
[Poster](/pdf/dirt_poster.pdf)

### RGB 6D Pose estimation for road vehicles
<a href='https://www.kaggle.com/c/pku-autonomous-driving'>Competition website</a>
<br>
Position: [45th/833]
<br>

### Diabetic Retionpathy Classification
<a href='https://www.kaggle.com/c/aptos2019-blindness-detection'>Competition website</a>
<br>
Position: [81st/2943]
<br>

[comment]: <> ([Project 3 Title]&#40;http://example.com/&#41;)

[comment]: <> (<img src="images/dummy_thumbnail.jpg?raw=true"/>)

[comment]: <> (---)

[comment]: <> (### Category Name 2)

[comment]: <> (- [Project 1 Title]&#40;http://example.com/&#41;)

[comment]: <> (- [Project 2 Title]&#40;http://example.com/&#41;)

[comment]: <> (- [Project 3 Title]&#40;http://example.com/&#41;)

[comment]: <> (- [Project 4 Title]&#40;http://example.com/&#41;)

[comment]: <> (- [Project 5 Title]&#40;http://example.com/&#41;)

[comment]: <> (---)




---
<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
