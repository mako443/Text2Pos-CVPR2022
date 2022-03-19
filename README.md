# Text2Pos: Text-to-point-cloud cross-modal localization
Code, models and KITTI360Pose dataset for our paper _"Towards cross-modal pose localization from text-based position descriptions"_ at CVPR 2022.
Main website: [text2pos.github.io](text2pos.github.io)

# Abstract
Natural language-based communication with mobile devices and home appliances is becoming increasingly popular and has the potential to become natural for communicating with mobile robots in the future. Towards this goal, we investigate cross-modal text-to-point-cloud localization that will allow us to specify, for example, a vehicle pick-up or goods delivery location. In particular, we propose Text2Pos, a cross-modal localization module that learns to align textual descriptions with localization cues in a coarse-to-fine manner. Given a point cloud of the environment, Text2Pos locates a position that is specified via a natural language-based description of the immediate surroundings. To train Text2Pos and study its performance, we construct KITTI360Pose, the first dataset for this task based on the recently introduced KITTI360 dataset. Our experiments show that we can localize $65\%$ of textual queries within $15m$ distance to query locations for top-10 retrieved locations. This is a starting point that we hope will spark future developments towards language-based navigation. 
    
# Dependencies
Our core dependencies are 
```
easydict
numpy
sklearn
matplotlib
cv2
open3d
torch
torch_geometric
```
with `pptk` as an optional package to visualize the point clouds.
Please pay close attention to the combined installation of CUDA, PyTorch and PyTorch Geometric as their versions are inter-connected and also depend on your available GPU. It might be required to install these packages by hand with specific versions instead of using the Pip-defaults.

# The KITTI360Pose dataset
We create KITTI360Pose from the [KITTI360](http://www.cvlibs.net/datasets/kitti-360/) dataset.
In particular, we sample 14,934 positions and generate up to three descriptions for each, totaling in 43,381 position-query pairs. We use five scenes (districts) for training (covering in total 11.59 km^2), one for model validation, and three for testing (covering in total 2.14 km^2). An average district covers an area of 1.78 km^2. 
Our baseline version of KITTI360Pose can be accessed here LINK.

## Create KITTI360Pose with custom parameters
After accessing the original KITTI360 and saving it under `./data/kitti360`, our KITTI360Pose baseline for a specific scene can be created using
```
python -m datapreparation.kitti360.prepare --scene_name 2013_05_28_drive_0000_sync --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
```
These parameters can be varied to create other setups of the dataset, as used in our ablation studies.

# Get started

## Downloading pre-trained models

## Run evaluation
After setting up the dependencies and dataset and downloading our pre-trained models, the baseline evaluation can be run with:
```
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_nm6/ \
--path_coarse ./checkpoints/coarse_contN_acc0.35_lr1_p256.pth \
--path_fine ./checkpoints/fine_acc0.88_lr1_obj-6-16_p256.pth 
```

## Train baseline models
After setting up the dependencies and dataset, our baseline models can be trained using the following commands:

```
python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_nm6/
python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_nm6/
```
