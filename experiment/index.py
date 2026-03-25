import heapq
from ipaddress import collapse_addresses
import time
from attr import dataclass
import numba
import veritas
import prada
import os
import numpy as np
import depth_first_ocenum
import h5py
import verification
import json
import util
import pickle


class Index:
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray):
        self.at = at
        self.featmap = featmap
        self.active_map = np.where(self.featmap != -1)[0]
        self.n_features = np.count_nonzero(self.featmap != -1)

    def _init_index(self):
        pass

    def store(self, boxes_buffer, outvalues_buffer):
        pass

    def to_arrays(self):
        pass

class IndexNode:

    def __init__(self, box, value=None):
        self.box = box
        self.children = []
        self.value = value
        self.pos = None
        self.neg = None

    def is_leaf(self):
        return len(self.children) == 0


class OCIndex(Index):
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray, depth: int = None):
        super().__init__(at, featmap)
        self.depth = depth if depth is not None else min(len(at) - 2, 10)
        self.init_index()
        self.build_index(self.depth)

    def init_index(self):
        self.index = []

    def build_index(self, depth):
        for d in range(1, depth + 1):
            for boxes_buffer, outvalues_buffer in depth_first_ocenum.enumerate_ocs_subset(self.at, d, 1000 * 8192):
                self.bulk_insert(self.index, boxes_buffer, outvalues_buffer)

    def bulk_insert(self, node_list, boxes, values):
        """
        node_list : list[OCNode]
        boxes     : (N, 2, D)
        values    : (N,)
        """
        if boxes.shape[0] == 0:
            return

        used = np.zeros(boxes.shape[0], dtype=bool)

        # Try to place boxes into existing nodes
        for node in node_list:
            mask = overlaps_batch(node.box, boxes)
            if not mask.any():
                continue

            used |= mask

        # Recurse once per node, not once per box
            self.bulk_insert(
                node.children,
                boxes[mask],
                values[mask],
            )

        # Boxes that didn't overlap any existing node → new nodes
        new_idx = np.where(~used)[0]
        for i in new_idx:
            n = IndexNode(boxes[i])
            n.value = values[i]
            node_list.append(n)
    def store(self, boxes_buffer, outvalues_buffer):
        self.bulk_insert(self.index, boxes_buffer, outvalues_buffer)

    def to_arrays(self):
        def collapse_leaves(node, n_features):
            """
            Converts lowest-level children of `node` into pos/neg arrays.
            Modifies the tree in place.
            """
            if not node.children:
                return False  # nothing to do here

            # Check if all children are leaves
            if all(len(child.children) == 0 for child in node.children):
                pos_boxes = []
                neg_boxes = []

                for child in node.children:
                    # each child may have multiple values
                    if child.value > 0:
                        pos_boxes.append(child.box)
                    else:
                        neg_boxes.append(child.box)

                # Allocate arrays once
                if pos_boxes:
                    pos = np.stack(pos_boxes).astype(np.float32)
                else:
                    pos = np.empty((0, 2, n_features), dtype=np.float32)

                if neg_boxes:
                    neg = np.stack(neg_boxes).astype(np.float32)
                else:
                    neg = np.empty((0, 2, n_features), dtype=np.float32)

                # Replace children with arrays
                node.children = None
                node.pos = pos
                node.neg = neg

                return True

            # Otherwise recurse deeper
            for child in node.children:
                collapse_leaves(child, n_features)

            return False
        for node in self.index:
            collapse_leaves(node, self.n_features)

    def get_boxes(self, x, label):
        box = x
        if x.ndim == 1:
            b = [x[f] for f in np.where(self.featmap != -1)[0]]
            box = np.array([b, b], dtype=np.float32)
        # Finds the boxes at the index that x belongs to
        def find_boxes(index, box, label):
            for node in index:
                if node.pos is not None and node.neg is not None:
                    if overlaps(node.box, box):
                        return node.pos if label == 1 else node.neg
                if overlaps(node.box, box):
                    return find_boxes(node.children, box, label)
                
        return find_boxes(self.index, box, label)

@numba.njit
def overlaps_batch(obox, boxes):
    # boxes: (N, 2, D)
    out = np.empty(boxes.shape[0], dtype=np.bool_)
    for i in range(boxes.shape[0]):
        ok = True
        for d in range(obox.shape[1]):
            if obox[0, d] > boxes[i, 0, d] or boxes[i, 1, d] > obox[1, d]:
                ok = False
                break
        out[i] = ok
    return out

class OCIndexSlow(Index):
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray, depth: int = None):
        super().__init__(at, featmap)
        self.depth = depth if depth is not None else len(at) - 2
        self._init_index()
        self.build_index(self.depth)

    def _init_index(self):
        self.index = []

    def build_index(self, depth):
        for boxes_buffer, outvalues_buffer in depth_first_ocenum.enumerate_ocs_subset(self.at, depth, 1000 * 8192):
            for box, outvalue in zip(boxes_buffer, outvalues_buffer):
                insert_index(self.index, box, outvalue)

    def to_arrays(self):
        def process_node(node):
            box, children = node
            # if children are all leaves -> convert children boxes to an array
            if all(isinstance(c, tuple) and type(c[1]) == np.float32 for c in children):
                pos = np.array([c[0] for c in children if c[1] > 0])
                neg = np.array([c[0] for c in children if c[1] <= 0])
                if pos.shape[0] == 0:
                    pos = np.empty((0, 2, self.n_features), dtype=np.float32)
                if neg.shape[0] == 0:
                    neg = np.empty((0, 2, self.n_features), dtype=np.float32)
                return (box, pos, neg)
            if len(children) == 0:
                return (box, np.empty((0, 2, self.n_features), dtype=np.float32), np.empty((0, 2, self.n_features), dtype=np.float32))
            
            # otherwise recurse deeper
            return (box, [process_node(c) for c in children])

        self.index = [process_node(n) for n in self.index]

    def store(self, boxes_buffer, outvalues_buffer):
        for j in range(boxes_buffer.shape[0]):
            box = boxes_buffer[j]
            pred = outvalues_buffer[j]
            insert_oc(self.index, box, pred)


    def get_boxes(self, x, label):
        box = x
        if x.ndim == 1:
            b = [x[f] for f in np.where(self.featmap != -1)[0]]
            box = np.array([b, b], dtype=np.float32)
        # Finds the boxes at the index that x belongs to
        def find_boxes(index, box, label):
            for node in index:
                if len(node) == 3:
                    obox, pos_children, neg_children = node
                    if overlaps(obox, box):
                        return pos_children if label == 1 else neg_children
                else:
                    obox, children = node
                    if overlaps(obox, box):
                        return find_boxes(children, box, label)
                
        return find_boxes(self.index, box, label)
    
    def get_children(self, index_entry):
        if len(index_entry) == 3:
            obox, pos_children, neg_children = index_entry
            return pos_children, neg_children
        else:
            obox, children = index_entry
            return [child for child in children]
    # def get_children(self, box):
    #     # Finds the boxes at the index that x belongs to
    #     return find_children(self.index, box)
    
def find_children(index, box):
    for node in index:
        if len(node) == 3:

            obox, pos_children, neg_children = node
            if equals(obox, box):
                return pos_children, neg_children
        else:
            obox, children = node
            if equals(obox, box):
                return [child[0] for child in children]
            if overlaps(obox, box):
                return find_children(children, box)

    


def insert_index(index, box, outvalue):
    for i in range(len(index)):
        if overlaps(index[i][0], box): #order important!
            if not equals(index[i][0], box):
                insert_index(index[i][1], box, outvalue)
            break
    else:
        index.append((box, []))

def insert_oc(index, box, outvalue):
    for i in range(len(index)):
        if overlaps(index[i][0], box): #order important!
            insert_oc(index[i][1], box, outvalue)
            break
    else:
        index.append((box, outvalue))

@numba.njit
def overlaps(obox, ibox):
    for d in range(obox.shape[1]):
        if obox[0, d] > ibox[0, d] or ibox[1, d] > obox[1, d]:
            return False
    return True

@numba.njit
def equals(obox, ibox):
    for d in range(obox.shape[1]):
        if obox[0, d] != ibox[0, d] or obox[1, d] != ibox[1, d]:
            return False
    return True

class PosNegIndex(Index):
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray):
        super().__init__(at, featmap)
        self._init_index()

    def _init_index(self):
        self.pos_index = []
        self.neg_index = []

    def store(self, boxes_buffer, outvalues_buffer):
        for j in range(boxes_buffer.shape[0]):
            box = boxes_buffer[j]
            pred = outvalues_buffer[j]
            if pred > 0:
                self.pos_index.append(box)
            else:
                self.neg_index.append(box)

    def get_boxes_at_index(self, label):
        if label == 1:
            return self.pos_index
        else:
            return self.neg_index

    def to_arrays(self):
        self.pos_index = np.array(self.pos_index, dtype=np.float32) if len(self.pos_index) > 0 else np.empty((0, 2, self.n_features), dtype=np.float32)
        self.neg_index = np.array(self.neg_index, dtype=np.float32) if len(self.neg_index) > 0 else np.empty((0, 2, self.n_features), dtype=np.float32)

class RootboxIndex(Index):
    # TODO replace rootbox index with flat encoding
    def __init__(self, at: veritas.AddTree, featmap: np.ndarray):
        super().__init__(at, featmap)
        self._define_splits()
        self._define_rootboxes()
        self._init_index()

        # Additional initialization code can go here

    def _define_splits(self):
        # Code to find the rootbox splits in the ensemble
        splits = {}
        for t in self.at:
            root_split = t.get_split(0) # Root box is at index 0
            if root_split.feat_id not in splits:
                splits[root_split.feat_id] = set()
            splits[root_split.feat_id].add(np.float32(root_split.split_value))

        self.splits = {k: np.array(sorted(list(v))) for k, v in sorted(splits.items())}

    def _define_rootboxes(self):
        self.nb_rootboxes = np.prod([len(v) + 1 for v in self.splits.values()])
        self.rootboxes = np.empty((self.nb_rootboxes, 2, len(self.at.get_splits().keys())), dtype=np.float32)
        self.rootboxes[:, 0, :] = -np.inf
        self.rootboxes[:, 1, :] = np.inf
        self.strides = {}
        for i, feat_id in enumerate(self.splits.keys()):
            self.strides[feat_id] = int(np.prod([len(self.splits[k]) + 1 for k in list(self.splits.keys())[i + 1:]]))

        # Iterate over all combinations of splits to define rootboxes

        feat_ids = list(self.splits.keys())
        intervals = {feat_id: [-np.inf] + list(self.splits[feat_id]) + [np.inf] for feat_id in feat_ids}
        for f, feat_id in enumerate(feat_ids): # Iterate over all used features in root splits
            j = self.featmap[feat_id]
            for i in range(self.nb_rootboxes):
                idx = (i // self.strides[feat_id]) % (len(self.splits[feat_id]) + 1) # Iterate over different ranges depending on the feature
                self.rootboxes[i, 0, j] = intervals[feat_id][int(idx)]
                self.rootboxes[i, 1, j] = intervals[feat_id][int(idx) + 1]


    def _init_index(self):
        self.index = {i: [[], []] for i in range(self.nb_rootboxes)}

    def get_nb_rootboxes(self):
        return self.nb_rootboxes
    
    def get_rootbox(self, index):
        return self.rootboxes[index]

    def get_rootboxes(self):
        return self.rootboxes

    def get_boxes_at_index(self, index, label):
        return self.index[index][label]
    
    def find_indices(self, boxes):
        # boxes: (N, D)
        X = boxes[:, 0, :]
        indices = np.zeros(X.shape[0], dtype=np.int64)

        for feat_id, split_vals in self.splits.items():
            col = X[:, self.featmap[feat_id]]
            counts = np.searchsorted(split_vals, col, side="right")
            indices += counts * self.strides[feat_id]

        return indices

    def find_index(self, iput: np.ndarray):
        # Finds the rootbox index that a given box/instance belongs to
        if iput.ndim == 1:
            x = iput[self.active_map]
        else:
            x = iput[0, :]

        index = 0
        # x = iput[0, :] if iput.ndim == 2 else iput # Take lower bounds in case of box input as all points in a box belong to the same rootbox
        for feat_id, values in self.splits.items():
            count = np.searchsorted(values, x[self.featmap[feat_id]], side='right')
            index += count * self.strides[feat_id]
        return int(index)
    
    def rootbox_generator(self, x):
        start = self.find_index(x)

        visited = set()
        heap = [(0.0, start)]

        while heap:
            dist, idx = heapq.heappop(heap)
            if idx in visited:
                continue
            visited.add(idx)

            yield dist, idx

            for nb_idx, cost in self._neighbors(idx, x):
                if nb_idx not in visited:
                    heapq.heappush(heap, (dist + cost, nb_idx))
        
    def _neighbors(self, index, x):
        for feat_id, values in self.splits.items():
            j = self.featmap[feat_id]
            interval_size = np.prod([len(self.splits[k]) + 1 for k in self.splits.keys() if k > feat_id])
            current_idx = (index // interval_size) % (len(values) + 1)

            # Check lower neighbor
            if current_idx > 0:
                nb_index = index - interval_size
                split_value = values[current_idx - 1]
                cost = max(0.0, split_value - x[j])
                yield (nb_index, cost)

            # Check upper neighbor
            if current_idx < len(values):
                nb_index = index + interval_size
                split_value = values[current_idx]
                cost = max(0.0, x[j] - split_value)
                yield (nb_index, cost)

    def store(self, boxes_buffer, outvalues_buffer):
        indices = self.find_indices(boxes_buffer)

        for j, i in enumerate(indices):
            if outvalues_buffer[j] > 0:
                self.index[i][1].append(boxes_buffer[j])
            else:
                self.index[i][0].append(boxes_buffer[j])
                
    # def store(self, boxes_buffer, outvalues_buffer):
    #     for j in range(boxes_buffer.shape[0]):
    #         box = boxes_buffer[j]
    #         pred = outvalues_buffer[j]
    #         i = self.find_index(boxes_buffer[j])
    #         # print('index_time:', time.time() - t)
    #         if pred > 0:
    #             self.index[i][1].append(box)
    #         else:
    #             self.index[i][0].append(box)
    #         # print('storing_time:', time.time() - t)
    def dump(self, oc_file):
        with h5py.File(oc_file, 'a') as f:
            group = f.create_group(f'rootbox_index')
            for i in range(self.nb_rootboxes):
                group.create_dataset(f'rootbox_{i}', data=np.array(self.index[i], dtype=np.int64), compression="gzip")
            f.attrs['feat_map'] = self.featmap
            f.attrs['at'] = self.at.to_json()

    def load(oc_file):
        with h5py.File(oc_file, 'r') as f:
            featmap = f.attrs['feat_map']
            at = veritas.AddTree.from_json(f.attrs['at'])

            rbi = RootboxIndex(at, featmap)
            for i in range(rbi.nb_rootboxes):
                rbi.index[i] = f[f'rootbox_index/rootbox_{i}'][:]
        return rbi
    
    def to_arrays(self):
        for i in range(self.nb_rootboxes):
            self.index[i][0] = np.array(self.index[i][0], dtype=np.float32) if len(self.index[i][0]) > 0 else np.empty((0, 2, self.n_features), dtype=np.float32)
            self.index[i][1] = np.array(self.index[i][1], dtype=np.float32) if len(self.index[i][1]) > 0 else np.empty((0, 2, self.n_features), dtype=np.float32)


if __name__ == "__main__":
    np.random.seed(5823)

    test_model_file = "testmodel.at"
    dname = "California"
    seed = 5823

    d = prada.get_dataset(dname, seed=seed, silent=False)
    d.load_dataset()
    d.robust_normalize()
    d.scale_target()
    d.astype(veritas.FloatT)

    if d.is_binary():
        d.use_balanced_accuracy()
    dtrain, dtest = d.train_and_test_fold(0, nfolds=4)

    if os.path.isfile(test_model_file):
        print("reading AddTree file...")
        at = veritas.AddTree.read(test_model_file, compressed=True)

    else:
        # Load dataset
        model_type = "xgb"
        model_class = d.get_model_class(model_type)

        # Fit XGB model
        params = {
            "random_state": seed,
            "n_jobs": 1,
            "n_estimators": 10,
            "max_depth": 4,
            "learning_rate": 1.0,
        }
        clf, _ = dtrain.train(model_class, params)

        # Transfer to Veritas AddTree
        at = veritas.get_addtree(clf)

        veritas.test_conversion(at, dtrain.X, clf.predict_proba(dtrain.X)[:, 1])
        print("writing file...")
        at.write(test_model_file, compressed=True)

    # splits = at.get_splits()
    # feat_ids = sorted(splits.keys())
    # num_feats = len(feat_ids)

    # feat_map = np.full(max(feat_ids)+1, -1, dtype=int)
    # for i, fid in enumerate(feat_ids):
    #     feat_map[fid] = i

    # print("building rootbox index...")
    # rbi = RootboxIndex(at, feat_map)
    # print(f"number of rootboxes: {rbi.get_nb_rootboxes()}")

    # oci = OCIndex(at, feat_map, depth=2)
    # print(oci.index)

    # _, enumeration_time, num_solutions, _ = depth_first_ocenum.enumerate_ocs(at, 'testmodel_ocs.h5', 1000 * 8192)



    # print(rbi.get_rootboxes())
    # index = rbi.find_index(np.array([[0.5, 0.5, 0]]))
    # print("rootbox index for box [[0.5, 0.5, 0]]:", index)
    # print("rootbox:", rbi.get_rootbox(index))

    with open('experiment/pareto_fronts_LOP.pkl', 'rb') as f:
        pareto_fronts = pickle.load(f)

    # Load all
    
    lop_compressed_models_on_front = {}
    with open('experiment/results/xgb_classification_saved.txt', 'r') as file: #Load in the XGB classification results
        for line in file:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            pareto_fronts_ds = pareto_fronts[line_dict['dname']]
            if pareto_fronts_ds[(pareto_fronts_ds['n_estimators'].astype(float) == float(line_dict['params']['n_estimators'])) &
                            (pareto_fronts_ds['max_depth'].astype(float) == float(line_dict['params']['max_depth'])) &
                            (pareto_fronts_ds['learning_rate'].astype(float) == float(line_dict['params']['learning_rate']))]['on_front'].values[0] == False:
                continue
            key = f"{line_dict['dname']}_{line_dict['params']['n_estimators']}_{line_dict['params']['max_depth']}_{line_dict['params']['learning_rate']}_{line_dict['fold']}"
            lop_compressed_models_on_front[key] = line_dict['refinements'][3]['model_json']
    
    oc_file = f'/cw/dtaiproj/ml/2026-OCspace/OC_enum_California_25_4_0.1_fold3.h5'

    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, 3, True)

    key = 'California_25_4_0.1_3'
    at = veritas.AddTree.from_json(lop_compressed_models_on_front[key])

    with h5py.File(oc_file, 'r') as f:
        print(f['boxes'])
        print(f['outvalues'])

    # print(verification.emp_robustness_rootbox_index(at, oc_file, dtest.X[:100].to_numpy(), dtest.y[:100].to_numpy(), 100000))
    results = verification.run_robustness_task(at, oc_file, dtest.X, dtest.y, 0.1, 0.15, 100000, 500)
    results1 = verification.run_verification_tasks(at, dtest.X, dtest.y, oc_file, 100000, 500)
    print(results)
    print(np.count_nonzero(np.array(results1['exact_emp_rob_robustness_values']) < 0.1))
    # print('robustness_results:', results['approx_emp_rob'], results['exact_emp_rob'], results['exact_emp_rob_rootbox_index'], results['exact_emp_rob_linear_scan'], results['exact_emp_rob_oc_index'])
    # # print(results)
    # print('verification_time_index:', np.sum(results['exact_emp_rob_rootbox_index_verification_times']))
    # print('verification_time_linear_scan:', np.sum(results['exact_emp_rob_linear_scan_verification_times']))
    # print('verification_time_kantchelian:', np.sum(results['exact_emp_rob_verification_times']))
    # print('verification_time_approx:', np.sum(results['approx_emp_rob_verification_times']))
    # print('verification_time_oc_index:', np.sum(results['exact_emp_rob_oc_index_verification_times']))

    # print('index time oc_index:', results['exact_emp_rob_oc_index_index_time'])
    # print('index time rootbox_index:', results['exact_emp_rob_rootbox_index_index_time'])
    # # print('reading_time_index:', results['exact_emp_rob_rootbox_index_reading_time'])
    # # print('reading_time_linear_scan:', results['exact_emp_rob_linear_scan_reading_time'])
    # print('diff:', [x - y for x, y in zip(results['exact_emp_rob_rootbox_index_robustness_values'], results['exact_emp_rob_robustness_values'])])
          



