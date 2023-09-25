# Semantics and NeRF: Towards 3D Semantic Scene Understanding from images

This blog covers the line of work in integrating semantic knowledge into NeRF (Neural Rendering Field) for 3D Semantic Scene Understanding. 

*disclaimer: I am not an author for any of the presented work*

## Background: [NeRF](https://www.matthewtancik.com/nerf) (ECCV 2020)
NeRF, or Neural Radiance Fields, gained major attention when it was introduced with the paper [*Representing Scenes as Neural Radiance Fields for View Synthesis*](https://www.matthewtancik.com/nerf) in ECCV 2020. (Perhaps this idea stems much further back but this is the earliest occasion I know that really began to gain attention).

The approach works by training a MLP for each individual 3D scene, where the MLP optimizes for "an underlying continous volumetric scene function". Formally put, the MLP learns function F where

$$F_{\Theta}(x,y,z, \theta, \phi) = RGB\sigma$$

Hence, the MLP as input takes the 3D xyz location (of the observed point) and the viewing point ($\theta, \phi$) as input, and outputs the corresponding color (RGB) and density($\sigma$) as output.

To train this network, it's important to note that NeRF utilizes **differential rendering** - using points along the camera rays as query points, NeRF is used to collect the RGB and radiance density along each ray to generate a rendering of the scene that is compared to the *actual image* provided by the data. Therefore, the loss is similar to a reconstruction loss where NeRF is optimized to produce renderings that best match the actual observation, and when trained under this objective NeRF may be queries from different view points for neural view synthesis.

## [In-Place Scene Labelling and Understanding with Implicit Scene Representation](https://shuaifengzhi.com/Semantic-NeRF/) (Semantic NeRF; ICCV 2021)

Naturally, when considering NeRF and semantics, one may consider the following question:

>  *Can NeRF be extended to be trained with Semantic labels?* 

And the answer is yes! Zhi et. al. was able to achieve this by extending the NeRF architecture to output semantic distribution $s$ in addition to the original RGB and emission density:

![semantic nerf]("images/semantic_nerf.png")

*Semantic NeRF architecture[1]*

As shown above, given dataset of RGB images with known poses, as well as their **semantic labels**, Semantic NeRF is trained to not only render the RGB viewpoint but also the corresponding segmentation labels. Therefore, the authors now have a 3D reconstruction of the scene with semantics.

Besides 3D semantic scene understanding, Semantic NeRF provides additional features such as denoising (learning 3D scene from multiple views allows model to render denoised semantic mask), super resolution (3D reconstruction learned from coarse/sparse label can transfer to finer resolution at inference), and full 3D scene reconstruction (NeRF has this capability but implicitly).

## [NeSF: Neural Semantic Fields for Generalizable Semantic Segmentation of 3D Scenes (TMLR 2022)](https://nesf3d.github.io/)

Exploring further from the idea initially explored by Semantic NeRF, Vora et. al. explore generalizable 3D semantic scene understanding by decoupling learning 3D geometry (orignial NeRF objective) and semantics (using semantic labels).

![nesf]("images/NeSF.png")

The key idea is to split the *geometric reconsturction* via NeRF and *semantic labeling* via 3D UNet. Upon training NeRF from set of images, the learned density field is then passed to the 3D UNet for segmentation. It's important to note here that then NeSF can be applied to unlabeled scenes, where only the RGB images are needed for NeRF, whereas the 3D UNet trained from other datasets can be used for semantic labeling. This allows generalization to novel scenes where the semantics are provided by pre-trained segmentation network, while 3D reconstruction is conducted by NeRF in self-supervised manner (since NeRF only needs the set of RGB images of the scene for implicit 3D view synthesis and density grid modelling).

![nesf]("images/NeSF_training.png")

The Semantic 3D UNet is also trained via differential rendering and does not require explicit 3D labels, and therefore the framework can learn from sparsely labeled data (i.e. set of RGB images for scenes with labels provided for only some of the images).


## [Panoptic Neural Fields: A semantic Object-Aware Neural Scene Representation](https://abhijitkundu.info/projects/pnf/) (CVPR 2022)

Panoptic Neural Fields developed by Kundu et. al. generalize beyond prior work by developing neural fields for dynamic outdoors scenes with panoptic capabilities, therefore able to detect not only the semantics but individual instances of the scene. Their key improvement is that they can *capture dynamic scenes* as well as *semantic instances*.

![nesf]("images/panoptic_nerf.png")

The key architectural design of their method is to train separate MLPs for stuff (terrain) and for things (objects) independently. While the stuff classes are trained with a foreground and background MLP, each instance of thing classes are trained by a separate MLP. Because each object instance is represented by a separate MLP, the networks can be very small in comparison to developing one very large MLP for the whole scene. Similar to prior, they use volumetric rendering generated by their MLPs to optimize network parameters with the observed RGB images and predicted 2D semantic images.

The learned representation at test time can then be used for various tasks - depth, instance segmentation, semantic segmentation, and RGB by generating rendering from the trained MLPs.

While their limitations may be that they do require a lot of prior information either provided or predicted i.e. camera poses, object tracks, semantic segmentations, they show remarkable results on outdoor scenes that is usually challenging for NeRF.

### References
[1] Zhi, Shuaifeng, et al. "In-place scene labelling and understanding with implicit scene representation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.