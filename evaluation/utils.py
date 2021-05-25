import numpy as np

def calc_sample_accuracies(pose, top_cells, pos_in_cells, top_k, threshs):
    pose_w = pose.pose_w
    assert len(top_cells) == max(top_k) == len(pos_in_cells)
    num_samples = len(top_cells)
    
    # Calc the pose-prediction in world coordinates for each cell
    pred_w = np.array([top_cells[i].bbox_w[0:2] + pos_in_cells[i, :] * top_cells[i].cell_size for i in range(num_samples)])

    # Calc the distances to the gt-pose
    dists = np.linalg.norm(pose_w[0:2] - pred_w, axis=1)
    assert len(dists) == max(top_k)

    # Discard close-by distances from different scenes
    pose_scene_name = pose.cell_id.split('_')[0]
    cell_scene_names = np.array([cell.id.split('_')[0] for cell in top_cells])
    dists[pose_scene_name != cell_scene_names] = np.inf

    # Calculate the accuracy: is one of the top-k dists small enough?
    return {k: {t: np.min(dists[0:k]) <= t for t in threshs} for k in top_k}

def depr_eval_pose_accuracies(dataset, retrievals, pos_in_cell, top_k=[1,3,5], threshs=[30,]):
    """
    Note: All done in 2D x-y / floor plane

    Args:
        dataset ([type]): [description]
        retrievals ([type]): [description]
        pos_in_cell ([type]): [description]
        top_k (list, optional): [description]. Defaults to [1,3,5].
        threshs (list, optional): [description]. Defaults to [30,].
    """
    assert len(dataset) == len(retrievals) # A retrieval for each query
    assert len(dataset) == len(pos_in_cell) # An offset prediction for each cell

    # WEITER: Send in poses as List of objects, re-project accordingly. 
    
    # For each cell, get the actual pose-prediciton in world-coordinates
    pose_preds = []
    pose_truths = []
    cell_scenes = []
    for idx in range(len(dataset)):
        # Project cell, pose and offset back to world-coordination
        bbox = dataset.cells[idx].bbox_w
        cell_size = dataset.cells[idx].cell_size
        pose_preds.append(bbox[0:2] + pos_in_cell[idx] * cell_size)
        pose_truths.append(dataset.cells[idx].pose_w[0:2])
        cell_scenes.append(dataset.cells[idx].scene_name)
    pose_preds, pose_truths, cell_scenes = np.array(pose_preds), np.array(pose_truths), np.array(cell_scenes)

    accuracies = {t: {k: [] for k in top_k} for t in threshs}
    for query_idx, sorted_indices in enumerate(retrievals):
        dists = np.linalg.norm(pose_truths[query_idx] - pose_preds[sorted_indices[0:np.max(top_k)]], axis=1) # Dists from gt-pose to top-retrievals
        dists[cell_scenes[query_idx] != cell_scenes[sorted_indices[0:np.max(top_k)]]] = np.inf # Remove retrievals from incorrect scenes

        for t in threshs:
            for k in top_k:
                accuracies[t][k].append(np.min(dists[0:k]) <= t)

    # Calculate mean accuracies
    for t in threshs:
        for k in top_k:
            accuracies[t][k] = np.mean(accuracies[t][k])

    return accuracies

def print_accuracies(accuracies, name=""):
    print(f'\n\t Accuracies {name}:')
    top_k = list(accuracies.keys())

    for t in accuracies[top_k[0]]:
        print(f'\t{t:4.0f}', end="")
    print()

    for k in accuracies:
        print(f'{k:2.0f}:\t', end="")
        for t in accuracies[k]:
            # print(f'{t:2.0f}: {accuracies[k][t]:0.2f}\t', end="")
            print(f'{accuracies[k][t]:0.2f}\t', end="")
        print()
    print("", flush=True)

def depr_print_accuracies(accuracies, name=""):
    print(f'\n\t Accuracies {name}:')
    for t in accuracies:
        print(f'{t:2.0f}:\t', end="")
        for k in accuracies[t]:
            print(f'{k:2.0f}: {accuracies[t][k]:0.2f}\t', end="")
        print()
    print("", flush=True)

        
