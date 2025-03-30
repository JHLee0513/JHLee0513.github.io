## Segment Every Out-of-Distribution Object
[[ArXiv]](https://arxiv.org/pdf/2311.16516.pdf)

* Meta-framework: it can utilize 1) an OOD scoring method to first generate segmentation mask of OOD detection score, 2) trained prompt generation model that detects bounding boxes given score map, 3) feed generated boxes to a promptable segmentation model such as SAM to get complete object/terrain-centric OOD mask
* The framework hence addreses the issuing arising the quality of OOD masks- incomplete, noisy mask when using SOTA OOD deteection methods.
* The authors only train the prompt generator, which is trained with a synthetically generated dataset via Outlier Exposure. i.e. using another seg dataset as OOD and one dataset as ID, masks from OOD dataset are randomly placed into the ID dataest as injected outliers.