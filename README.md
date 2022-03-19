# Text2Pos: Text-to-point-cloud cross-modal localization
Code, models and KITTI360Pose dataset for our paper _"Towards cross-modal pose localization from text-based position descriptions"_ at CVPR 2022.
Main website: [text2pos.github.io](text2pos.github.io)

# Abstract
Natural language-based communication with mobile devices and home appliances is becoming increasingly popular and has the potential to become natural for communicating with mobile robots in the future. Towards this goal, we investigate cross-modal text-to-point-cloud localization that will allow us to specify, for example, a vehicle pick-up or goods delivery location. In particular, we propose Text2Pos, a cross-modal localization module that learns to align textual descriptions with localization cues in a coarse-to-fine manner. Given a point cloud of the environment, Text2Pos locates a position that is specified via a natural language-based description of the immediate surroundings. To train Text2Pos and study its performance, we construct KITTI360Pose, the first dataset for this task based on the recently introduced KITTI360 dataset. Our experiments show that we can localize $65\%$ of textual queries within $15m$ distance to query locations for top-10 retrieved locations. This is a starting point that we hope will spark future developments towards language-based navigation. 
    
# Dependencies

# The KITTI360Pose dataset

## Create KITTI360Pose with custom parameters

# Get started

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
