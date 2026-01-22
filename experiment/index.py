from attr import dataclass
import numba
import veritas
import prada
import os
import numpy as np
import depth_first_ocenum
import h5py
import verification


@dataclass
class Index:
    at: veritas.AddTree
    featmap: np.ndarray
    index: dict

    def _init_index(self):
        pass
        



class RootboxIndex(Index):
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

        self.splits = {k: sorted(list(v)) for k, v in sorted(splits.items())}

    def _define_rootboxes(self):
        self.nb_rootboxes = np.prod([len(v) + 1 for v in self.splits.values()])
        self.rootboxes = np.empty((self.nb_rootboxes, 2, len(self.at.get_splits().keys())), dtype=np.float32)
        self.rootboxes[:, 0, :] = -np.inf
        self.rootboxes[:, 1, :] = np.inf

        # Iterate over all combinations of splits to define rootboxes

        feat_ids = list(self.splits.keys())
        intervals = {feat_id: [-np.inf] + self.splits[feat_id] + [np.inf] for feat_id in feat_ids}
        for f, feat_id in enumerate(feat_ids): # Iterate over all used features in root splits
            j = self.featmap[feat_id]
            for i in range(self.nb_rootboxes):
                idx = (i // np.prod([len(self.splits[feat_ids[k]]) + 1 for k in range(f + 1, len(feat_ids))])) % (len(self.splits[feat_id]) + 1) # Iterate over different ranges depending on the feature
                self.rootboxes[i, 0, j] = intervals[feat_id][int(idx)]
                self.rootboxes[i, 1, j] = intervals[feat_id][int(idx) + 1]


    def _init_index(self):
        self.index = {i: [] for i in range(self.nb_rootboxes)}


    def get_nb_rootboxes(self):
        return self.nb_rootboxes
    
    def get_rootbox(self, index):
        return self.rootboxes[index]

    def get_rootboxes(self):
        return self.rootboxes

    def get_boxes_at_index(self, index):
        return self.index[index]
    
    def find_index(self, iput: np.ndarray):
        # Finds the rootbox index that a given box/instance belongs to
        x = iput[0, :] if iput.ndim == 2 else iput # Take lower bounds in case of box input as all points in a box belong to the same rootbox
        for feat_id, values in self.splits.items():
            for i in range(0, len(values)):
                if x[self.featmap[feat_id]] >= values[i]:
                    index += np.prod([len(self.splits[k]) + 1 for k in sorted(self.splits.keys()) if k > feat_id])
        return int(index)
    
    def store(self, start_idx, buffer):
        for j, box in enumerate(buffer):
            i = self.find_index(box)
            self.index[i].append(start_idx + j)

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

if __name__ == "__main__":
    test_model_file = "testmodel.at"
    dname = "Phoneme"
    seed = 12

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
            "n_estimators": 17,
            "max_depth": 5,
            "learning_rate": 1.0,
        }
        clf, _ = dtrain.train(model_class, params)

        # Transfer to Veritas AddTree
        at = veritas.get_addtree(clf)

        veritas.test_conversion(at, dtrain.X, clf.predict_proba(dtrain.X)[:, 1])
        print("writing file...")
        at.write(test_model_file, compressed=True)

    splits = at.get_splits()
    feat_ids = sorted(splits.keys())
    num_feats = len(feat_ids)

    feat_map = np.full(max(feat_ids)+1, -1, dtype=int)
    for i, fid in enumerate(feat_ids):
        feat_map[fid] = i

    print("building rootbox index...")
    rbi = RootboxIndex(at, feat_map)
    print(f"number of rootboxes: {rbi.get_nb_rootboxes()}")

    _, enumeration_time, num_solutions = depth_first_ocenum.enumerate_ocs(at, 'testmodel_ocs.h5', 100 * 8192)
    # print("enumeration took", time.time() - start_time, "seconds")
    print(f"number of solutions: {num_solutions}")
    print(f"enumeration time: {enumeration_time} seconds")
    rbi = RootboxIndex.load('testmodel_ocs.h5')

    # print(rbi.get_rootboxes())
    # index = rbi.find_index(np.array([[0.5, 0.5, 0]]))
    # print("rootbox index for box [[0.5, 0.5, 0]]:", index)
    # print("rootbox:", rbi.get_rootbox(index))

    results = verification.run_verification_tasks(at, dtest.X, dtest.y, 'testmodel_ocs.h5', 100000, 500)

    print('robustness_results:', results['approx_emp_rob'], results['exact_emp_rob'], results['exact_emp_rob_rootbox_index'], results['exact_emp_rob_linear_scan'])
    print(results)
    print('verification_time_index:', np.sum(results['exact_emp_rob_rootbox_index_verification_times']))
    print('verification_time_linear_scan:', np.sum(results['exact_emp_rob_linear_scan_verification_times']))
    print('verification_time_kantchelian:', np.sum(results['exact_emp_rob_verification_times']))
    print('verification_time_approx:', np.sum(results['approx_emp_rob_verification_times']))
    print('diff:', [x - y for x, y in zip(results['exact_emp_rob_rootbox_index_robustness_values'], results['exact_emp_rob_robustness_values'])])
          



