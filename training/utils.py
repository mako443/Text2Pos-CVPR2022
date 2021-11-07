import numpy as np
import cv2
from easydict import EasyDict
from numpy.lib.arraysetops import isin
from dataloading.kitti360.base import Kitti360BaseDataset
from datapreparation.kitti360.drawing import plot_cell, plot_matches_in_best_cell

import torch_geometric.transforms as T

def set_border(img, color):
    img[0:15, :] = color
    img[-15:, :] = color
    img[:, 0:15] = color
    img[:, -15:] = color

def plot_matches(matches0, dataset, count=20):
    print(len(matches0), len(dataset))
    print(matches0[0])
    assert isinstance(matches0, list) and len(matches0) == len(dataset)

    for _ in range(count):
        idx = np.random.randint(len(dataset))
        data = dataset[idx]
        pose = data['poses']
        cell = data['cells']
        pred_matches = matches0[idx]
        gt_matches = data['matches']

        img = plot_matches_in_best_cell(cell, pose, pred_matches, gt_matches)
        cv2.imwrite(f'matches_{idx}.png', img)
        print(f'Idx {idx} saved!')    

def plot_retrievals(top_retrievals, dataset, count=10, top_k=3, green_thresh=10):
    assert isinstance(top_retrievals, list)
    
    cells_dict = {cell.id: cell for cell in dataset.all_cells}

    count_plotted = 0
    while count_plotted < count:
        idx = np.random.randint(len(top_retrievals))
        pose = dataset.all_poses[idx]
        retrievals = top_retrievals[idx]
        # if pose.cell_id in retrievals[0 : top_k]:

        images_semantic_rgb = []
        for use_rgb in (False, True):
            images = []
            # Add query
            images.append(plot_cell(cells_dict[pose.cell_id], use_rgb=use_rgb, pose=pose.pose))

            # Add top-k
            for cell_id in retrievals[0 : top_k]:
                cell = cells_dict[cell_id]
                img = plot_cell(cell, use_rgb=use_rgb)

                # Add border and distance
                dist = np.linalg.norm(pose.pose_w[0:2] - cell.get_center()[0:2])
                border_color = (0, 255, 0) if pose.scene_name == cell.scene_name and dist < green_thresh else (0, 0, 255)
                set_border(img, border_color)

                h, w, d = img.shape
                cv2.rectangle(img, (25, h-125), (img.shape[1]//5 * 3, h-20), (255, 255, 255), thickness=-1)
                dist_text = 'n/a' if pose.scene_name != cell.scene_name else f'{dist:0.2f}'
                cv2.putText(img, 'Distance: ' + dist_text, (50, img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), thickness=2)
                
                images.append(img)

            sep = np.ones((images[0].shape[0], 100, 3), np.uint8) * 255
            images.insert(1, sep)
            images_semantic_rgb.append(np.hstack(images))
        
        cv2.imwrite(f"ret_{count_plotted}.png", np.vstack(images_semantic_rgb))
        print('Saved pos!')
        count_plotted += 1

    # # Plot <count> negatives
    # count_neg = 0 
    # while count_neg < count:
    #     idx = np.random.randint(len(top_retrievals))
    #     pose = dataset.all_poses[idx]
    #     retrievals = top_retrievals[idx]
    #     if pose.cell_id not in retrievals[0 : top_k]:
    #         images = []
    #         images.append(plot_cell(cells_dict[pose.cell_id]))
    #         for cell_id in retrievals[0 : top_k]:
    #             images.append(plot_cell(cells_dict[cell_id]))

    #         sep = np.ones((images[0].shape[0], 100, 3), np.uint8) * 255
    #         images.insert(1, sep)
    #         cv2.imwrite(f"ret_neg_{count_neg}.png", np.hstack(images))
    #         print('Saved neg!')
    #         count_neg += 1

def depr_plot_retrievals(top_retrievals, dataset, count=5):
    """Plots <count> targets and their top-3 retrievals next to each other if the target is not in top-10.
    CARE: Assumes query and database items have the same indices, i.e. query-index == target-index.

    Args:
        top_retrievals (dict): Top-retrievals as {query_idx: [sorted_indices]}
        dataset (Dataset): Training or validation dataset
        count (int, optional): How many plots to create. Defaults to 5.
    """
    raise Exception('Not updated to query_idx != target_idx!')
    
    query_indices = list(top_retrievals.keys())
    query_indices = np.random.choice(query_indices, size=len(query_indices), replace=False)

    indices_plotted = 0
    for query_idx in query_indices:
        retrieval_indices = top_retrievals[query_idx]
        if query_idx in retrieval_indices[0:10]: # Only plot retrievals where the target is not in top-10
            continue

        img0 = plot_cell(dataset.cells[query_idx], scale=512)
        img1 = plot_cell(dataset.cells[retrieval_indices[0]], scale=512)
        img2 = plot_cell(dataset.cells[retrieval_indices[1]], scale=512)
        img3 = plot_cell(dataset.cells[retrieval_indices[2]], scale=512)

        sep = np.ones((img0.shape[0],10,3), dtype=np.uint8) * 255

        img = np.hstack((img0, sep, img1, img2, img3))
        cv2.imwrite(f'retrievals_{query_idx}.png', img)

        print(f'Plotted {query_idx}: {retrieval_indices[0]} {retrieval_indices[1]} {retrieval_indices[2]}')

        indices_plotted += 1
        if indices_plotted >= count:
            break

if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    
    
    dataset = Kitti360BaseDataset(base_path, folder_name)
    top_retrievals = {0: [0,1,2], 1: [0,1,2], 2: [0,1,2]}

    plot_retrievals(top_retrievals, dataset, count=3)