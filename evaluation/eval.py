# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from scipy.spatial.transform import Rotation as R

# from datapreparation.imports import calc_angle_diff

# def eval_pose_predictions(data_set, model, thresholds=[(10, np.pi/8), (15, np.pi/4), (20, np.pi/2)]):
#     for i in range(len(thresholds)):
#         thresholds[i] = (thresholds[i][0], np.float16(thresholds[i][1]))

#     data_loader = DataLoader(data_set, batch_size=10, num_workers=2, pin_memory=False, shuffle=False) #CARE: no shuffle

#     pose_predictions = np.zeros((0,6))
#     with torch.no_grad():
#         for batch in data_loader:
#             out = model(batch['descriptions'])
#             out = out.detach().cpu().numpy()
#             pose_predictions = np.vstack((pose_predictions, out))

#     threshold_hits = {t: [] for t in thresholds}
#     for i in range(len(pose_predictions)):
#         key = data_set.poses_keys[i]
#         pos_diff = np.linalg.norm(data_set.poses[key].eye[0:2] - pose_predictions[i, 0:2])
#         ori_diff = calc_angle_diff(data_set.poses[key].phi, R.from_quat(pose_predictions[i, 2:6]).as_rotvec()[-1])

#         for thresh in thresholds:
#             if pos_diff <= thresh[0] and ori_diff <= thresh[1]:
#                 threshold_hits[thresh].append(True)
#             else:
#                 threshold_hits[thresh].append(False)

#     for thresh in thresholds:
#         threshold_hits[thresh] = np.float16(np.mean(threshold_hits[thresh]))

#     return threshold_hits