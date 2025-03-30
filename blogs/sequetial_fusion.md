# Sequential Fusion



## Remaining external questions

Evidently, LiDAR is still a very expensive sensor to put in commercial platforms at a large scale, and unlike sensors such as radar has not been extensively developed mounting on vehicles as well. 

## Example Study: KeypointDet

As a followup to the survey, I explore a method related but quite different from decorating pointclouds or fusing LiDAR with additional sensor data. In light of camera-based 3D object detection, this project explores the use of camera and depth estimation to generate saliant keypoints that serve as a pointcloud input into the LiDAR-based detection architectures.

### Method:
3 stage network:
1. off-the-shelf depth estimation is used to get rough analysis of the geometry of the data
2. candidate mask generation, an image semantic segmentation network is used to generate ROIs
3. keypoint generation: for each ROI, the keypoint network is passed in to generate set of N keypoints, which then uses interpolated depth estimation to generate a point cloud input
4. object detection: the generated pointcloud is then used to detect the objects with an existing 3D LiDAR architecture.