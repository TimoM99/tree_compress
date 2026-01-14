import os
os.environ['PRADA_DATA_DIR']='/cw/dtaijupiter/NoCsBack/dtai/timo/prada_data'
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
    ("_lids", numba.int64[:]),
    ("_outvalues", numba.float32[:]),
    ("_base_score", numba.float32),
])
class NumbaAddTreeBoxes(object):
    def __init__(self, feat_map, offset, los, his, lvals, base_score):
        self.feat_map = feat_map
        self.offset = offset
        self.los = los
        self.his = his
        self.lvals = lvals
        self._workspace = np.zeros((len(offset), 2, los.shape[1]), dtype=np.float32)
        self._lids = np.zeros(len(offset)-1, dtype=np.int64)
        self._outvalues = np.zeros(len(offset), dtype=np.float32)
        self._base_score = base_score

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
        self._lids[:] = 0
        self._outvalues[:] = 0.0
        self._outvalues[0] = self._base_score

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
        boxes.feat_map,
        offsets,
        stacked_los,
        stacked_his,
        stacked_lvals,
        boxes.at.get_base_score(0),
    )
    return nboxes


@numba.njit
def intersect_lo(los_in1, los_in2, los_out):
    for i in range(len(los_in1)):
        los_out[i] = max(los_in1[i], los_in2[i])


@numba.njit
def intersect_hi(his_in1, his_in2, his_out):
    for i in range(len(his_in1)):
        his_out[i] = min(his_in1[i], his_in2[i])


@numba.njit
def overlaps(los0, his0, los1, his1):
    overlap = True  # intentionally branchless
    for l0, h0, l1, h1 in zip(los0, his0, los1, his1):
        overlap &= (l0 <= h1) & (h0 > l1)
    return overlap


@numba.njit
def enumerate_ocs_recursive(nboxes, tree_index, outvalue):
    ws0 = nboxes._workspace[tree_index, :, :]
    lvals = nboxes.get_lvals(tree_index)

    if tree_index < nboxes.num_trees():
        los, his = nboxes.get_lohis(tree_index)
        num_leaves = los.shape[0]  # == his.shape[0]
        ws1 = nboxes._workspace[tree_index+1, :, :]

        for lid in range(num_leaves):
            if not overlaps(ws0[0, :], ws0[1, :], los[lid, :], his[lid, :]):
                continue
            intersect_lo(ws0[0, :], los[lid, :], ws1[0, :])
            intersect_hi(ws0[1, :], his[lid, :], ws1[1, :])

            enumerate_ocs_recursive(nboxes, tree_index+1, outvalue+lvals[lid])

    else:
        print(ws0, "→", outvalue)


@numba.njit
def enumerate_ocs_stack(nboxes, tree_index, box_buffer, outvalue_buffer):
    buffer_index = 0
    num_trees = nboxes.num_trees()
    while buffer_index < box_buffer.shape[0]:
        if tree_index >= num_trees:
            print("fail")
            break

        ws0 = nboxes._workspace[tree_index, :, :]
        lid = nboxes._lids[tree_index]
        los, his = nboxes.get_lohis(tree_index)
        lvals = nboxes.get_lvals(tree_index)
        num_leaves = los.shape[0]

        ws1 = nboxes._workspace[tree_index + 1, :, :]

        if lid >= num_leaves:  # all leaves of this tree were considered
            if tree_index == 0:  # we are done!
                break
            nboxes._lids[tree_index] = 0  # start over again with this tree in the next round
            tree_index -= 1  # backtrack to previous tree
            continue

        nboxes._lids[tree_index] += 1

        if overlaps(ws0[0, :], ws0[1, :], los[lid, :], his[lid, :]):
            intersect_lo(ws0[0, :], los[lid, :], ws1[0, :])
            intersect_hi(ws0[1, :], his[lid, :], ws1[1, :])

            outvalue = nboxes._outvalues[tree_index] + lvals[lid]
            nboxes._outvalues[tree_index]
            if tree_index == num_trees - 1:  # solution, no more next tree to move to
                #print(ws1, "→", outvalue, "solution", buffer_index)
                box_buffer[buffer_index, 0, :] = ws1[0, :]
                box_buffer[buffer_index, 1, :] = ws1[1, :]
                outvalue_buffer[buffer_index] = outvalue
                buffer_index += 1
            else:
                tree_index += 1  # next iteration moves to the next tree
                nboxes._outvalues[tree_index] = outvalue

    return tree_index, buffer_index









def enumerate_ocs(at):
    splits = at.get_splits()
    feat_ids = sorted(splits.keys())
    num_feats = len(feat_ids)

    feat_map = np.full(max(feat_ids)+1, -1, dtype=int)
    for i, fid in enumerate(feat_ids):
        feat_map[fid] = i
    print("feat_map", feat_map, "num used features", num_feats, "max feature id", max(feat_ids))

    boxes = AddTreeBoxes(at, feat_map)

    # Define the boxes of all leaves on every tree
    for m, t in enumerate(at):
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)

        # Keep track of the leaf values as well
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
    enumerate_ocs_recursive(nboxes, 0, at.get_base_score(0))

    print()
    print("NON RECURSIVE")

    tree_index = 0
    buffer_size = 10
    box_buffer = np.zeros((buffer_size, 2, num_feats), dtype=np.float32)
    outvalue_buffer = np.zeros(buffer_size, dtype=np.float32)
    nboxes.reset_workspace()

    while True:
        box_buffer[:, :, :] = 0.0
        outvalue_buffer[:] = 0.0
        tree_index, num_solutions = enumerate_ocs_stack(
            nboxes, tree_index, box_buffer, outvalue_buffer
        )

        # do something with the stuff in the buffer (e.g. write to file, index, ...)
        print(outvalue_buffer[:num_solutions])
        print(box_buffer[:num_solutions, :, :])

        if num_solutions < buffer_size:  # we're done, nothing more was written to the buffer
            break



if __name__ == "__main__":
    test_model_file = "testmodel.at"
    dname = "Phoneme"
    seed = 12

    if os.path.isfile(test_model_file):
        print("reading AddTree file...")
        at = veritas.AddTree.read(test_model_file, compressed=True)

    else:
        # Load dataset
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

        # Transfer to Veritas AddTree
        at = veritas.get_addtree(clf)

        veritas.test_conversion(at, dtrain.X, clf.predict_proba(dtrain.X)[:, 1])
        print("writing file...")
        at.write(test_model_file, compressed=True)

    enumerate_ocs(at)

