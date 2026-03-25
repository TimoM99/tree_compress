import json
import random
import time
from turtle import distance
from matplotlib import defaultParams
import veritas
import numpy as np
from numba import njit
import util

@njit
def min_dist_to_solutions(example, target_label, all_inds, all_doms, preds, n_intervals):
    min_dist = 1e18
    for s in range(n_intervals):
        max_dist = 0.0
        inds = all_inds[s]
        doms = all_doms[s]
        pred = preds[s]
        if (pred > 0.0) != target_label:
            continue
        for i in range(len(inds)):
            idx = inds[i]
            if idx == -1:
                break
            x = example[idx]
            low, high = doms[i, 0], doms[i, 1]
            if x < low:
                d = low - x
            elif x > high:
                d = x - high
            else:
                d = 0.0
            if d > max_dist:
                max_dist = d
        if max_dist < min_dist:
            min_dist = max_dist
    return min_dist

@njit
def compute_dist_boxes_point(point, doms, inds):
    result = np.empty(doms.shape[0], dtype=np.float32)
    for i in range(doms.shape[0]):
        dom = doms[i][inds[i] != -1]
        ind = inds[i][inds[i] != -1]

        min_point = point.copy()
        min_point[ind] = np.clip(min_point[ind], dom[:, 0], dom[:, 1])

        # max_point = point.copy()
        # max_point[ind] = np.where(np.abs(max_point[ind] - dom[:, 1]) > np.abs(max_point[ind] - dom[:, 0]), dom[:, 1], dom[:, 0])

        m = np.max(np.abs(min_point - point))
        # M = np.max(np.abs(max_point - point))

        result[i] = m
        # result[i, 1] = M

    return result

@njit
def find_point_in_box(dom, ind, point, find_min):

    result = point.copy()
    if find_min:
        result[ind] = np.clip(point[ind], dom[0, :], dom[1, :])
    else:
        raise ValueError("Max point computation not implemented yet")

    return result

@njit
def dist(a, b, type):
    match type:
        case 'l_inf':
            return l_inf_dist(a, b)
        case 'l2':
            return l2_dist(a, b)
        case 'l1':
            return l1_dist(a, b)
        case _:
            raise ValueError(f"Unknown distance type: {type}")
    
@njit
def l_inf_dist(a, b):
    return np.max(np.abs(a - b))

@njit
def l2_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

@njit
def l1_dist(a, b):
    return np.sum(np.abs(a - b))

@njit
def find_delta(example, distance, target_label, all_inds, all_doms, preds, index):
    # timer_start = time.time()
    # target_doms = all_doms[(preds > 0) == target_label]
    # target_inds = all_inds[(preds > 0) == target_label]
    idx = np.searchsorted(index, distance, side='right')
    delta = min_dist_to_solutions(example, target_label, all_inds[:idx], all_doms[:idx], preds[:idx], idx)
    # dists = compute_dist_boxes_point(example, target_doms[:idx], target_inds[:idx])
    # delta_alt = np.min(dists) if len(dists) > 0 else distance
    # assert delta == delta_alt

    idx_2 = np.searchsorted(index, distance + delta, side='right')
    delta_2 = min_dist_to_solutions(example, target_label, all_inds[idx:idx_2], all_doms[idx:idx_2], preds[idx:idx_2], idx_2 - idx)
    # dists = compute_dist_boxes_point(example, target_doms[idx:idx_2], target_inds[idx:idx_2])
    # delta_2 = np.min(dists) if len(dists) > 0 else delta

    # print('done')
    # return 0, 0.0
    return min(delta, delta_2), idx_2/len(index)

class OCIndex:
    def __init__(self, at, doms, inds, preds):
        self.at = at
        self.doms = doms
        self.inds = inds
        self.preds = preds
        self.sorted = False
        
        self.ref_point = np.zeros(at.get_maximum_feat_id(), dtype=np.float32)
        self.index = compute_dist_boxes_point(self.ref_point, doms, inds)
        
    def dist_to_ref(self, x):
        return l_inf_dist(x[:len(self.ref_point)], self.ref_point)
    
    def count_boxes_within_dist(self, d):
        assert self.sorted, "Index must be sorted before counting boxes within distance"
        return np.searchsorted(self.index, d, side='right')

    def find_delta(self, example, target_label):
        assert self.sorted, "Index must be sorted before checking robustness"
        distance = self.dist_to_ref(example)
        return find_delta(example, distance, target_label, self.inds, self.doms, self.preds, self.index)

    def sort_index(self):
        sorted_index = self.index.argsort()
        self.doms = self.doms[sorted_index]
        self.inds = self.inds[sorted_index]
        self.preds = self.preds[sorted_index]
        self.index = self.index[sorted_index]
        self.sorted = True




# Make a weighted count of feature usage across all trees in an ensemble
def analyze_feature_usage(at):
    feature_counts = {}
    for t in at:

        def traverse_tree(t, node):
            if t.is_leaf(node):
                return
            split = t.get_split(node)
            feature_id = split.feat_id
            
            if feature_id not in feature_counts:
                feature_counts[feature_id] = 0
            
            feature_counts[feature_id] += 2**(t.max_depth() - t.depth(node))

            traverse_tree(t, t.left(node))
            traverse_tree(t, t.right(node))

        traverse_tree(t, 0)
    return {k: v for k, v in sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)}

def check_robustness(at, x, y, n, index):
    count = 0
    delta_total = 0.0   
    pct_checked_total = 0.0
    t = time.time()
    for i in x.index:
        target_label = not (y.loc[i] > 0.0)
        example = x.loc[i, :].to_numpy().astype(np.float32)
        pred_label = at.eval(example)[0, 0] > 0.0

        if pred_label != target_label:

            delta, pct_checked = index.find_delta(example, target_label)
            delta_total += delta
            pct_checked_total += pct_checked

        count += 1
        if count >= n:
            break
        # if time.time() - t > timeout:
        #     break
    t = time.time() - t
    return delta_total / count, pct_checked_total / count, t

# dname = 'Volkert[2v7]'
# fold = 2
# n_estimators = 100
# max_depth = 6
# learning_rate = 0.5

dname = 'Vehicle'
fold = 0
n_estimators = 25
max_depth = 4
learning_rate = 0.5

file_xgb = f"experiment/results/xgb_classification_saved.txt"
with open(file_xgb, "r") as f:
    for line in f:
        if not line.startswith('{'):
            continue
        line_dict = json.loads(line.strip())
        if line_dict['dname'] != dname or line_dict['fold'] != fold or line_dict['params']['n_estimators'] != n_estimators or line_dict['params']['max_depth'] != max_depth or line_dict['params']['learning_rate'] != learning_rate:
            continue

        else: 
            model = veritas.AddTree.from_json(line_dict['refinements'][3]['model_json'])
            assert line_dict['refinements'][3]['penalty'] == 'ours'

            break

print(model.num_leafs())
print(analyze_feature_usage(model))
doms = np.load(f'/cw/dtailocal/timo/OCs/OC_boxes_{dname}_{line_dict["params"]["n_estimators"]}_{line_dict["params"]["max_depth"]}_{line_dict["params"]["learning_rate"]}_fold{fold}.npy', mmap_mode='r')
inds = np.load(f'/cw/dtailocal/timo/OCs/OC_inds_{dname}_{line_dict["params"]["n_estimators"]}_{line_dict["params"]["max_depth"]}_{line_dict["params"]["learning_rate"]}_fold{fold}.npy', mmap_mode='r')
preds = np.load(f'/cw/dtailocal/timo/OCs/OC_preds_{dname}_{line_dict["params"]["n_estimators"]}_{line_dict["params"]["max_depth"]}_{line_dict["params"]["learning_rate"]}_fold{fold}.npy', mmap_mode='r')

oc_index = OCIndex(model, doms, inds, preds)
print('index built')
oc_index.sort_index()
print('index sorted')
seed = util.SEED
np.random.seed(seed)
random.seed(seed)

d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent=True)

print(check_robustness(model, dtest.X, dtest.y, 500, oc_index))




print('works')
