import os
import veritas
import prada
import numpy as np
import numba

from dataclasses import dataclass, field


@dataclass
class AddTreeBoxes:
    at: veritas.AddTree
    feat_map: np.ndarray
    los: list[np.ndarray] = field(default_factory=list)
    his: list[np.ndarray] = field(default_factory=list)
    lvals: list[np.ndarray] = field(default_factory=list)


@numba.experimental.jitclass([
    ("feat_map", numba.int64[:]),
    ("offset", numba.int64[:]),
    ("los", numba.float32[:,::1]),
    ("his", numba.float32[:,::1]),
    ("lvals", numba.float32[:]),
    ("_workspace", numba.float32[:,:,::1]),
])
class NumbaAddTreeBoxes(object):
    def __init__(self, feat_map, offset, los, his, lvals):
        self.feat_map = feat_map
        self.offset = offset
        self.los = los
        self.his = his
        self.lvals = lvals
        self._workspace = np.zeros((len(offset), 2, los.shape[1]), dtype=np.float32)

    def get_lohis(self, tree_index: int):
        offset0 = self.offset[tree_index]
        offset1 = self.offset[tree_index+1]

        los = self.los[offset0:offset1, :]
        his = self.his[offset0:offset1, :]

        return los, his

    def get_lvals(self, tree_index: int):
        offset0 = self.offset[tree_index]
        offset1 = self.offset[tree_index+1]

        return self.lvals[offset0:offset1]

    def reset_workspace(self):
        self._workspace[:, 0, :] = -np.inf
        self._workspace[:, 1, :] = np.inf

    def num_trees(self):
        return len(self.offset) - 1



def create_numba_addtree_boxes(boxes):
    stacked_los = np.vstack(boxes.los)
    stacked_his = np.vstack(boxes.his)
    stacked_lvals = np.hstack(boxes.lvals)

    offsets = np.zeros(len(boxes.at)+1, dtype=np.int64)
    for i in range(1, len(boxes.at)):
        offsets[i] = offsets[i-1] + boxes.los[i].shape[0]
    offsets[-1] = stacked_lvals.shape[0]

    nboxes = NumbaAddTreeBoxes(
        boxes.feat_map, offsets, stacked_los, stacked_his, stacked_lvals
    )
    return nboxes


@numba.njit
def testnumba(nboxes):
    print("hello", nboxes.offset[1])

    print(nboxes.get_lohis(0)[0])
    print(nboxes.get_lohis(0)[1])
    print("jaja", nboxes.get_lvals(0))

@numba.jit
def intersect_lo(los_in1, los_in2, los_out):
    for i in range(len(los_in1)):
        los_out[i] = max(los_in1[i], los_in2[i])

@numba.jit
def intersect_hi(his_in1, his_in2, his_out):
    for i in range(len(his_in1)):
        his_out[i] = min(his_in1[i], his_in2[i])


@numba.njit
def enumerate_ocs_recursive(nboxes, tree_index):
    ws0 = nboxes._workspace[tree_index, :, :]

    if tree_index < nboxes.num_trees():
        los, his = nboxes.get_lohis(tree_index)
        ws1 = nboxes._workspace[tree_index+1, :, :]

        for lid in range(los.shape[0]):
            #print(lid, ws0[0, :].shape, los[lid, :].shape)
            intersect_lo(ws0[0, :], los[lid, :], ws1[0, :])
            intersect_hi(ws0[1, :], his[lid, :], ws1[1, :])

            #print(ws1)
            enumerate_ocs_recursive(nboxes, tree_index+1)

    else:
        print(ws0)











def enumerate_ocs(at):
    splits = at.get_splits()

    feat_map = np.full(max(splits.keys())+1, -1, dtype=int)
    for i, fid in enumerate(sorted(splits.keys())):
        feat_map[fid] = i
    num_feats = len(feat_map)

    boxes = AddTreeBoxes(at, feat_map)

    for m, t in enumerate(at):
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)

        lvals = np.array([t.get_leaf_value(lid, 0) for lid in leaf_ids], dtype=np.float32)
        los = np.full((num_leaves, num_feats), -np.inf, dtype=np.float32)
        his = np.full_like(los, np.inf)

        for i, lid in enumerate(leaf_ids):
            box = t.compute_box(lid)
            for fid, ival in box.items():
                los[i, feat_map[fid]] = ival.lo
                his[i, feat_map[fid]] = ival.hi

        boxes.los.append(los)
        boxes.his.append(his)
        boxes.lvals.append(lvals)

    nboxes = create_numba_addtree_boxes(boxes)
    nboxes.reset_workspace()
    enumerate_ocs_recursive(nboxes, 0)





if __name__ == "__main__":
    test_model_file = "testmodel.at"
    dname = "Phoneme"
    seed = 12

    if os.path.isfile(test_model_file):
        print("reading AddTree file...")
        at = veritas.AddTree.read(test_model_file, compressed=True)

    else:
        d = prada.get_dataset(dname, seed=seed, silent=False)
        d.load_dataset()
        d.robust_normalize()
        d.scale_target()
        d.astype(veritas.FloatT)

        if d.is_binary():
            d.use_balanced_accuracy()

        dtrain, dtest = d.train_and_test_fold(0, nfolds=4)

        model_type = "xgb"
        model_class = d.get_model_class(model_type)

        # Fit XGB model
        params = {
            "random_state": seed,
            "n_jobs": 1,
            "n_estimators": 2,
            "max_depth": 2,
            "learning_rate": 1.0,
        }
        clf, _ = dtrain.train(model_class, params)
        at = veritas.get_addtree(clf)

        veritas.test_conversion(at, dtrain.X, clf.predict_proba(dtrain.X)[:, 1])
        print("writing file...")
        at.write(test_model_file, compressed=True)

    enumerate_ocs(at)

