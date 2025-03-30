## EfficientNetV2: Smaller Models and Faster Training
[Paper](https://arxiv.org/abs/2104.00298)

Building upon EfficientNet, it introduces a set of design principles that result in improvement not only in efficiency (accuracy, parameter count), but also training speed.

1. NAS is augmented to include model training time in its reward, hence the search finds architecture optimized not only for model accuruacy but also fast training time.
2. Progressively increasing image size has been empirically known as an effective wayto train models faster early on, but larger image sizes also tend to cause overfitting. To mitigate this, authors devise progressive learning with adaptive regulariztion where augmentation e.g. fixmatch sample rate is increased alongside iamge size to support adapative reguarlization.
3. In the base architecture of EffNet, reserachers have found that using regular 3x3 conv instead of separable (depthwise conv + 1x1 conv) conv block within the MBConv(Mobile-Bottleneck Conv) block makes much better use of acceleartors and is therefore more runtime optimized design. However, authors have found that still having the default MBConv(mobile) at later stages is much more effective, while the exact cutoff to switch between the two is ambiguous. Therefore, authors  include both blocks in their NAS search to find that cutoff in learning fashion.
4. Authors found that the simple compound scaling rule from the original EfficientNet family is sub-optimal, and therefore adds a set of manually set heuristics (max image size to 480, more layers gradually added to stage 5,6).
5. Note that NAS is still only used to search for B0.
6. The original intuition behind compound scaling was that width, length, and image size all have to scale(grow/shrink) together in a principle way for the best tradeoff in scale. Therefore, a certain compound coefficient phi was defined to uniformly scale nework width, depth, and resolution.