import numpy as np
import cv2
from dataloading.kitti360.base import Kitti360BaseDataset
from datapreparation.kitti360.drawing import plot_cell

def plot_retrievals(top_retrievals, dataset, count=5):
    """Plots <count> targets and their top-3 retrievals next to each other if the target is not in top-10.
    CARE: Assumes query and database items have the same indices, i.e. query-index == target-index.

    Args:
        top_retrievals (dict): Top-retrievals as {query_idx: [sorted_indices]}
        dataset (Dataset): Training or validation dataset
        count (int, optional): How many plots to create. Defaults to 5.
    """
    query_indices = list(top_retrievals.keys())
    query_indices = np.random.choice(query_indices, size=len(query_indices), replace=False)

    indices_plotted = 0
    for query_idx in query_indices:
        retrieval_indices = top_retrievals[query_idx]
        if query_idx not in retrieval_indices[0:3]:
            continue

        img0 = plot_cell(dataset.cells[query_idx], scale=512)
        img1 = plot_cell(dataset.cells[retrieval_indices[0]], scale=512)
        img2 = plot_cell(dataset.cells[retrieval_indices[1]], scale=512)
        img3 = plot_cell(dataset.cells[retrieval_indices[2]], scale=512)

        img = np.hstack((img0, img1, img2, img3))
        cv2.imwrite(f'retrievals_{query_idx}.png', img)

        indices_plotted += 1
        if indices_plotted >= count:
            break

if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    
    
    dataset = Kitti360BaseDataset(base_path, folder_name)
    top_retrievals = {0: [0,1,2], 1: [0,1,2], 2: [0,1,2]}

    plot_retrievals(top_retrievals, dataset, count=3)