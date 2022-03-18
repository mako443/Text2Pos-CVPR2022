from itertools import accumulate
import numpy as np
import os
import os.path as osp
import pickle
import cv2

from evaluation.utils import print_accuracies

'''
TODO
- Perform sanity-check between first db-image and next query-images

- Show retrievals
- Check with smaller distance
- Train model?!
'''

def evaluate(poses_db, poses_query, features_db, features_query, top_k=(1, 3, 5), thresh=(30, 60, 90)):
    assert len(poses_db) == len(features_db) and len(poses_query) == len(features_query)
    accuracies = {k: {t: [] for t in thresh} for k in top_k}

    retrievals = {} # {query_idx: [db_indices]}

    for query_idx in range(len(poses_query)):
        pose_dists = np.linalg.norm(poses_db - poses_query[query_idx], axis=1)
        feature_dists = np.linalg.norm(features_db - features_query[query_idx], axis=1)
        indices = np.argsort(feature_dists) # Low -> High

        retrievals[query_idx] = indices[0 : 3]

        for k in top_k:
            for t in thresh:
                topk_dists = pose_dists[indices[0 : k]]
                accuracies[k][t].append(np.min(topk_dists) <= t)
        
    for k in top_k:
        for t in thresh:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return accuracies, retrievals

# NOTE / CARE: The image indices might be wrong because real images are copied with 1-based indexing
def plot_retrievals(retrievals, path_images, count=5):
    rows = []
    for query_idx in np.random.randint(len(retrievals), size=count):
        db_indices = retrievals[query_idx]

        query_image = cv2.imread(osp.join(path_images, 'query', f'{query_idx:04.0f}.png'))
        db_images = [cv2.imread(osp.join(path_images, 'db', f'{idx:04.0f}.png')) for idx in db_indices]
        row_images = [query_image, ]
        for image in db_images:
            row_images.append(255 * np.ones((query_image.shape[0], 25, 3), dtype=np.uint8))
            row_images.append(image)
        rows.append(np.hstack(row_images))
    
    return np.vstack(rows)

if __name__ == '__main__':
    if True: # Vis.-loc. vs. cross-modal
        scene_name = '2013_05_28_drive_0010_sync'
        path = osp.join('./data', 'k360_30-10_scG_pd10_pc4_spY_all', 'visloc', scene_name)
        with open(osp.join(path, 'features_db.pkl'), 'rb') as f:
            features_db = pickle.load(f)
        with open(osp.join(path, 'features_query.pkl'), 'rb') as f:
            features_query = pickle.load(f)

        with open(osp.join(path, 'db', 'poses.pkl'), 'rb') as f:
            poses_db = pickle.load(f)

        with open(osp.join(path, 'query', 'poses.pkl'), 'rb') as f:
            poses_query = pickle.load(f) 

        accuracies, retrievals = evaluate(poses_db, poses_query, features_db, features_query, top_k=(1, 5, 10), thresh=(5, 10, 15))
        print('NetVLAD on free poses:')
        print_accuracies(accuracies)
        print()   
        
        quit()

    path = osp.join('./data', 'k360-visloc_db-25_q5', '2013_05_28_drive_0010_sync')
    image_type = 'real'
    
    for image_type in ('real', 'rendered'):
        with open(osp.join(path, f'features_{image_type}_db.pkl'), 'rb') as f:
            features_db = pickle.load(f)
        with open(osp.join(path, f'features_{image_type}_query.pkl'), 'rb') as f:
            features_query = pickle.load(f)            

        with open(osp.join(path, f'poses_db.pkl'), 'rb') as f:
            poses_db = pickle.load(f)

        with open(osp.join(path, f'poses_query.pkl'), 'rb') as f:
            poses_query = pickle.load(f)            

        # accuracies, retrievals = evaluate(poses_db, poses_db, features_db, features_db)

        # plot = plot_retrievals(retrievals, osp.join(path, image_type))
        # cv2.imwrite(f'retrievals-{image_type}.png', plot)

        accuracies, retrievals = evaluate(poses_db, poses_query, features_db, features_query, top_k=(1, 3, 5), thresh=(30, 60, 90))
        print('TYPE: ', image_type)
        print_accuracies(accuracies)
        print()  

        accuracies, retrievals = evaluate(poses_db, poses_query, features_db, features_query, top_k=(1, 5, 10), thresh=(5, 10, 15))
        print('TYPE: ', image_type)
        print_accuracies(accuracies)
        print()  
