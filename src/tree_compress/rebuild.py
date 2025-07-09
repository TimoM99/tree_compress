import numpy as np
import veritas
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression, Lasso

@dataclass
class Candidate:
    tree_index: int
    node_id: int


class Rebuild:
    def __init__(self, data, at, metric):
        self.seed = 46
        self.tol = 1e-5

        self.d = data
        self.at_orig = at

        self.at = veritas.AddTree(1, veritas.AddTreeType.REGR)
        for i in range(len(at)):
            self.at.add_tree()

        self.metric = metric
        self.mtrain_ref, self.mvalid_ref, self.mtest_ref = self._get_metrics(at)
        self.mtrain, self.mvalid, self.mtest = self._get_metrics(self.at)

        self.candidates = [Candidate(m, t.root()) for m, t in enumerate(at)]

    def is_regression(self):
        return False

    def _predat(self, at, x):
        """ Predict hard labels for veritas.AddTree """
        if self.is_regression():
            return at.predict(x)
        elif at.num_leaf_values() == 1:
            return (at.eval(x)[:, 0] >= 0.0).astype(int)
        else:
            return at.eval(x).argmax(axis=1)

    def _get_metrics(self, at):
        mtrain = self.metric(self.d.ytrain, self._predat(at, self.d.xtrain))
        mvalid = self.metric(self.d.yvalid, self._predat(at, self.d.xvalid))
        mtest = self.metric(self.d.ytest, self._predat(at, self.d.xtest))
        return mtrain, mvalid, mtest

    def _compute_gain(self, cand):
        t_orig = self.at_orig[cand.tree_index]
        split = t_orig.get_split(cand.node_id)
        atc = self.at.copy()
        t = atc[cand.tree_index]

        path = []
        n = cand.node_id
        while not t_orig.is_root(n):
            path.append(t_orig.is_left_child(n))
            n = t_orig.parent(n)

        n = t.root()
        for go_left in reversed(path):
            n = t.left(n) if go_left else t.right(n)

        leaf_value = t.get_leaf_value(n, 0)
        t.split(n, split.feat_id, split.split_value)
        t.set_leaf_value(t.left(n), 0, leaf_value)
        t.set_leaf_value(t.right(n), 0, leaf_value)

        self._fit_leaves(atc)

        mtrain, mvalid, mtest = self._get_metrics(atc)
        gain = mtrain - self.mtrain
        return atc, gain

    def _fit_leaves(self, at):
        num_rows = self.d.xtrain.shape[0]
        num_cols = sum(t.num_leaves() for t in at if not t.is_leaf(t.root()))
        xxtrain = np.zeros((num_rows, num_cols))
        offset = 0

        onetwothree = np.arange(num_rows)

        for m, t in enumerate(at):
            if t.is_leaf(t.root()):
                continue
            leaf_ids = t.get_leaf_ids()
            leaf_map = np.full(max(leaf_ids)+1, -1, dtype=int)
            for i, lid in enumerate(leaf_ids):
                leaf_map[lid] = i
            node_ids = leaf_map[t.eval_node(self.d.xtrain)] + offset
            xxtrain[onetwothree, node_ids] = 1.0
            offset += t.num_leaves()

        clf = self._fit_coefficients(at, xxtrain, self.d.ytrain)

        coef = clf.coef_[0]
        at.set_base_score(0, clf.intercept_[0])
        for m, t in enumerate(at):
            if t.is_leaf(t.root()):
                continue

            for i, lid in enumerate(t.get_leaf_ids()):
                t.set_leaf_value(lid, 0, coef[i])

    def _fit_coefficients(self, at, xx, y):
        clf = LogisticRegression(
            fit_intercept=True,
            #penalty="l1",
            #C=1.0,
            #solver="liblinear",
            max_iter=5_000,
            tol=self.tol,
            n_jobs=1,
            random_state=self.seed,
            warm_start=False,
        )
        clf.fit(xx, y)
        return clf

    def take_one_step(self):
        """Loop over all candidate new nodes and add the best one."""
        print(f"ref {self.mtrain_ref*100:5.1f}, {self.mvalid_ref*100:5.1f}")

        max_at = None
        max_candi = -1
        max_gain = -np.inf

        for i, cand in enumerate(self.candidates):
            at, gain = self._compute_gain(cand)
            if gain > max_gain:
                max_gain = gain
                max_candi = i
                max_at = at

            print(" - ", cand.tree_index, cand.node_id, f"{gain:.3f}")

        if max_at is not None:
            self.at = max_at
            cand = self.candidates[max_candi]
            t = self.at_orig[cand.tree_index]

            left = Candidate(cand.tree_index, t.left(cand.node_id))
            right = Candidate(cand.tree_index, t.right(cand.node_id))

            print(cand)

            self.candidates[max_candi] = left
            self.candidates.append(right)

        self.mtrain, self.mvalid, self.mtest = self._get_metrics(self.at)
        print(f"    {self.mtrain*100:5.1f}, {self.mvalid*100:5.1f}")

