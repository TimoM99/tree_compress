import itertools
import time
import gc
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import veritas
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.linear_model import Lasso, LogisticRegression

from .util import Data, at_isregr, at_predlab, count_nnz_leafs, print_fit, print_metrics


@dataclass
class CompressRecord:
    """
    Records data related to the compression of a tree ensemble.

    Attributes
    ----------
    level : int
        The compression level.
    at : veritas.AddTree
        The additive tree associated with this record.
    tindex : float
        Time taken for indexing, initialized to 0.0.
    ttransform : float
        Time taken for transformation, initialized to 0.0.
    tsearch : float
        Time taken for search, initialized to 0.0.
    ntrees : int
        Number of trees in the ensemble.
    nnodes : int
        Number of nodes in the ensemble.
    nleafs : int
        Number of leaf nodes in the ensemble.
    nnz_leafs : int
        Number of non-zero leaf nodes in the ensemble.
    mtrain : float
        Score on training data, initialized to 0.0.
    mtest : float
        Score on testing data, initialized to 0.0.
    mvalid : float
        Score on validation data, initialized to 0.0.
    alphas : list
        List of alpha values for regularization.
    clf_mtrain : list
        Metrics on training data for classification tasks.
    clf_mvalid : list
        Metrics on validation data for classification tasks.
    """

    level: int
    at: veritas.AddTree
    tindex: float = 0.0
    ttransform: float = 0.0
    tsearch: float = 0.0
    ntrees: int = field(init=False)
    nnodes: int = field(init=False)
    nleafs: int = field(init=False)
    nnz_leafs: int = field(init=False)
    mtrain: float = 0.0
    mtest: float = 0.0
    mvalid: float = 0.0
    alphas: List[float] = field(default_factory=list)
    clf_mtrain: List[float] = field(default_factory=list)
    clf_mvalid: List[float] = field(default_factory=list)

    def __post_init__(self):
        """
        Initializes derived attributes based on the provided AddTree object.
        """
        self.ntrees = len(self.at)
        self.nnodes = self.at.num_nodes()
        self.nleafs = self.at.num_leafs()
        self.nnz_leafs = count_nnz_leafs(self.at)


@dataclass
class AlphaRecord:
    lo: float
    hi: float
    alpha: float

    clf_mtrain: float = 0.0
    clf_mvalid: float = 0.0
    num_params: int = 0
    num_removed: int = 0
    num_kept: int = 0
    frac_removed: float = 0.0
    fit_time: float = 0.0

    intercept: Optional[np.ndarray] = None
    coefs: Optional[np.ndarray] = None


class AlphaSearch:
    def __init__(self, round_nsteps, mtrain_ref, mvalid_ref, isworse_fun):
        """Search for a regularizion strength parameter."""
        self.round_nsteps = round_nsteps
        self.mtrain_ref = mtrain_ref
        self.mvalid_ref = mvalid_ref
        self.isworse_fun = isworse_fun

        self.round = 0
        self.step = 1
        self.lo = -3
        self.hi = 4

        self.records = []

        self.set_lohis()

    def __iter__(self):
        return self

    def __next__(self):
        nsteps = self.nsteps()
        if self.step > nsteps:  # next round
            lo, hi = self.next_lohis()

            self.lo, self.hi = lo, hi
            self.set_lohis()
            self.step = 1

        lo, mid, hi = self.lohis[self.step - 1 : self.step + 2]
        alpha = np.power(10.0, mid)

        record = AlphaRecord(lo, hi, alpha)
        self.records.append(record)

        self.step += 1
        return record

    def isworse_tr(self, mtrain):
        return self.isworse_fun(mtrain, self.mtrain_ref)

    def isworse_va(self, mvalid):
        return self.isworse_fun(mvalid, self.mvalid_ref)

    def isnotworse_tr(self, mtrain):
        return not self.isworse_fun(mtrain, self.mtrain_ref)

    def isnotworse_va(self, mvalid):
        return not self.isworse_fun(mvalid, self.mvalid_ref)

    def overfits(self, mtrain, mvalid):
        cond1 = self.isnotworse_tr(mtrain)
        cond2 = self.isworse_va(mvalid)
        cond3 = self.isworse_fun(mvalid, mtrain)
        return cond1 and cond2 and cond3

    def underfits(self, mtrain, mvalid):
        cond1 = self.isworse_tr(mtrain)
        cond2 = self.isworse_va(mvalid)
        return cond1 and cond2

    def nsteps(self):
        return self.round_nsteps[self.round]

    def set_lohis(self):
        nsteps = self.nsteps()
        self.lohis = np.linspace(self.lo, self.hi, nsteps + 2)

    def quality_filter(self, records):
        filt = filter(
            lambda r: self.isnotworse_va(r.clf_mvalid)
            and r.frac_removed < 1.0
            and not self.isworse_fun(r.clf_mtrain, r.clf_mvalid),  # overfitting
            records,
        )
        return filt

    def next_lohis(self):
        nsteps = self.nsteps()
        num_rounds = len(self.round_nsteps)

        self.round += 1
        if self.round >= num_rounds:
            raise StopIteration()

        prev_round_records = self.records[-1 * nsteps :]

        # Out of the records whose validation metric is good enough...
        filt = self.quality_filter(prev_round_records)
        # ... pick the one with the highest alpha
        best = max(filt, default=None, key=lambda r: r.alpha)

        # If nothing was good enough, find the last record that was overfitting and the
        # first that was underfitting, and look at that transition in more detail
        r_over, r_under = None, None
        if best is None:
            for r in prev_round_records:
                if self.overfits(r.clf_mtrain, r.clf_mvalid):
                    r_over = r
                elif r_over is not None and self.underfits(r.clf_mtrain, r.clf_mvalid):
                    r_under = r
                    break
            if r_over is not None and r_under is not None:
                lo = (r_over.lo + r_over.hi) / 2.0
                hi = (r_under.lo + r_under.hi) / 2.0
                return lo, hi
            else:
                raise StopIteration()
        else:
            return best.lo, best.hi

    # def get_best_record(self):
    #     # Out of the good enough solutions...
    #     filt = self.quality_filter(self.records)
    #     return max(filt, default=None, key=lambda r: r.frac_removed)
    def get_best_record(self):
        m = max(self.quality_filter(self.records), default=None, key=lambda r: r.frac_removed)
        if m is None:
            return None
        
        allm = [r for r in self.quality_filter(self.records) if r.frac_removed == m.frac_removed]
        return allm[-1]


class Compress:
    def __init__(
        self,
        data: Data,
        at: veritas.AddTree,
        score: Callable[[np.ndarray, np.ndarray], float],
        isworse: Callable[[float, float], bool],
        silent: bool = False,
        seed: int = 569,
        fit_intercept: bool = True,
    ):
        self.d = data
        self.silent = silent
        self.no_convergence_warning = silent
        self.seed = seed
        self.fit_intercept = fit_intercept
        self.alpha_search_round_nsteps = [8, 4, 4]

        self.score = score
        self.isworse = isworse

        self.mtrain = self.score(self.d.ytrain, at_predlab(at, self.d.xtrain))
        self.mtest = self.score(self.d.ytest, at_predlab(at, self.d.xtest))
        self.mvalid = self.score(self.d.yvalid, at_predlab(at, self.d.xvalid))
        if not self.silent:
            print(
                "MODEL PERF:",
                f"mtr {self.mtrain:.3f} mte {self.mtest:.3f} mva {self.mvalid:.3f}",
            )

        self.records = [
            CompressRecord(
                level=-1,
                at=at,
                mtrain=self.mtrain,
                mtest=self.mtest,
                mvalid=self.mvalid,
            )
        ]

        self.at = at
        self.nlv = at.num_leaf_values()

        if self.nlv > 1:  # multi-target or multiclass
            self.at_singletarget = [at.make_singleclass(k) for k in range(self.nlv)]
        else:  # single target or binary classification
            self.at_singletarget = [at]

        self.mtrain_fortarget = [
            self.score(
                self._transformy(k, self.d.ytrain), at_predlab(at, self.d.xtrain)
            )
            for k in range(self.nlv)
        ]
        self.mtest_fortarget = [
            self.score(self._transformy(k, self.d.ytest), at_predlab(at, self.d.xtest))
            for k in range(self.nlv)
        ]
        self.mvalid_fortarget = [
            self.score(
                self._transformy(k, self.d.yvalid), at_predlab(at, self.d.xvalid)
            )
            for k in range(self.nlv)
        ]

    def is_regression(self):
        return at_isregr(self.at)

    # def _get_indexes(self, level, include_higher_leaves):
    #     return [
    #         self._get_index_fortarget(k, level, include_higher_leaves)
    #         for k in range(self.nlv)
    #     ]

    def _get_index_fortarget(self, target, level, include_higher_leaves):
        index = []  # tree_index -> node_idx -> col_idex
        num_cols = 0  # if 1, one intercept column of all ones

        for t in self.at_singletarget[target]:
            index0 = {}  # leaf_id -> node_id at level
            index1 = {}  # node_id at level -> xxmatrix column index

            # make an index: leaf_id -> [nodes along root-to-leaf path]
            for leaf_id in t.get_leaf_ids():
                n = leaf_id
                path = [n]
                while not t.is_root(n):
                    n = t.parent(n)
                    path.insert(0, n)

                # if this leaf is higher up than the level, either just take the leaf,
                if include_higher_leaves:
                    n_at_level = path[level] if len(path) > level else path[-1]

                # ... or do not include it this time.
                else:
                    n_at_level = path[level] if len(path) > level else None

                if n_at_level is not None:
                    index0[leaf_id] = n_at_level

                    if n_at_level not in index1:
                        index1[n_at_level] = num_cols
                        if t.is_root(n_at_level):
                            num_cols += 1
                        elif t.is_leaf(n_at_level):
                            num_cols += 1
                        else:
                            num_cols += 2

            index.append((index0, index1))
        return index, num_cols

    # def _transformx(self, x, indexes):
    #    blocks = []
    #    for k in range(self.nlv):
    #        index, num_cols = indexes[k]
    #        at = self.at_singletarget[k]
    #        xxk = self._transformx_fortarget(at, x, index, num_cols)
    #        blocks.append(xxk)
    #    nb = len(blocks)

    #    recipe = np.full((nb, nb), None, dtype=object)
    #    for k in range(nb):
    #        recipe[k, k] = blocks[k]

    #    xx = bmat(recipe, format=blocks[0].getformat())

    #    frac_nnz = xx.nnz / np.prod(xx.shape) if num_cols > 0 else 1.0
    #    if frac_nnz > 0.01:
    #        xx = xx.toarray()
    #    return xx

    def _transformx(self, at, x, index, num_cols):
        num_rows = x.shape[0]
        
        # Use direct dense matrix construction to avoid memory overhead
        # of building lists and converting from sparse matrices
        xx = np.zeros((num_rows, num_cols), dtype=np.float64, order="C")
        
        # AddTree transformation - fill matrix directly
        for m, t, (index0, index1) in zip(range(len(at)), at, index):
            leaf_ids = t.eval_node(x)  # Get all leaf IDs at once
            for i, leaf_id in enumerate(leaf_ids):
                if leaf_id not in index0:
                    continue

                leaf_value = t.get_leaf_value(leaf_id, 0)
                n_at_level = index0[leaf_id]
                col_idx = index1[n_at_level]
                
                if t.is_root(n_at_level) or t.is_leaf(n_at_level):
                    if t.is_root(n_at_level):
                        # a coefficient * leaf values for root nodes, no bias
                        xx[i, col_idx] = leaf_value
                    else:
                        xx[i, col_idx] = 1.0  # only bias term for leaves
                else:
                    xx[i, col_idx] = 1.0
                    xx[i, col_idx + 1] = leaf_value

        return xx 
    

    def _transformy(self, target, y):
        # if self.is_regression():
        #    if y.ndim == 2:
        #        return np.hstack([y[:, i] for i in range(self.nlv)])
        #    return y
        # else:
        #    ymat = self.y_encoder.transform(y.reshape(-1, 1)).toarray()
        #    ystacked = np.hstack([ymat[:, i] for i in range(self.nlv)])
        #    return ystacked
        if self.is_regression():
            if y.ndim == 2:
                return y[:, target]
            else:
                assert target == 0
                return y
        elif self.nlv == 1:  # binary classification
            assert target == 0
            return (y >= 0.5).astype(int)
        else:  # multiclass classification
            return (y == target).astype(int)

    #TODO fix that it works without timeout as well
    def compress(self, *, max_rounds=2, timeout=7200):
        timer = time.time()
        time_spent = 0.0
        last_record = self.records[-1]
        for i in range(max_rounds):
            if not self.silent:
                print(f"\n\nROUND {i+1}")

            self._compress_round(timeout=timeout - time_spent)

            new_record = self.records[-1]
            has_improved = new_record.nnodes < last_record.nnodes
            

            time_spent = time.time() - timer
            if not has_improved:
                break
            last_record = new_record
            # print(last_record.at.get_type())
        return last_record.at

    def _compress_round(self, timeout):
        timer = time.time()
        for level in itertools.count():
            if time.time() - timer > timeout:
                break
            r = self.compress_level(level)

            if not self.silent:
                r0 = self.records[0]
                r1 = self.records[-1]
                print_metrics("orig", r0)
                print_metrics("prev", r1, rcmp=r0, cmp=self.isworse)
                print_metrics("now", r, rcmp=r0, cmp=self.isworse)

                print("tr", self.mtrain_fortarget)
                print("te", self.mtest_fortarget)
                print("va", self.mvalid_fortarget)

                for target in range(self.nlv):
                    at_target = self.at_singletarget[target]
                    print(
                        target,
                        ":",
                        np.array(
                            [
                                self.score(
                                    self._transformy(target, self.d.ytrain),
                                    at_predlab(at_target, self.d.xtrain),
                                )
                                - self.mtrain_fortarget[target],
                                self.score(
                                    self._transformy(target, self.d.ytest),
                                    at_predlab(at_target, self.d.xtest),
                                )
                                - self.mtest_fortarget[target],
                                self.score(
                                    self._transformy(target, self.d.yvalid),
                                    at_predlab(at_target, self.d.xvalid),
                                )
                                - self.mvalid_fortarget[target],
                            ]
                        ),
                    )
                    print()

                print()

            self.records.append(r)

            # max_depth = max(
            #     max(tree.depth(i) for i in tree.get_leaf_ids()) for tree in self.at
            # )
            max_depth = self.at.max_depth()
            if level >= max_depth:
                if not self.silent:
                    print(f"DONE, depth of tree reached {level}, {max_depth}")
                break

        return self.records[-1].at

    def compress_level(self, level):
        bests = []
        new_full_at = self._new_empty_addtree(self.nlv)
        tindex = 0.0
        ttransform = 0.0
        tsearch = 0.0

        for target in range(self.nlv):
            t = time.time()
            index, num_cols = self._get_index_fortarget(target, level, True)
            tindex += time.time() - t
            at = self.at_singletarget[target]

            t = time.time()
            xxtrain = self._transformx(at, self.d.xtrain, index, num_cols)
            yytrain = self._transformy(target, self.d.ytrain)
            xxvalid = self._transformx(at, self.d.xvalid, index, num_cols)
            yyvalid = self._transformy(target, self.d.yvalid)
            ttransform += time.time() - t

            alpha_search = AlphaSearch(
                self.alpha_search_round_nsteps,
                self.mtrain_fortarget[target],
                self.mvalid_fortarget[target],
                self.isworse,
            )
            clf = self._get_regularized_lin_clf(xxtrain)

            tsearch = time.time()
            for i, alpha_record in enumerate(alpha_search):
                self._update_lin_clf_alpha(clf, alpha_record.alpha)
                clf = self.fit_coefficients(
                    clf, xxtrain, yytrain, xxvalid, yyvalid, alpha_record
                )

                if not self.silent:
                    print_fit(alpha_record, alpha_search)


            tsearch = time.time() - tsearch

            best = alpha_search.get_best_record()

            if best is not None:
                intercept = best.intercept
                coefs = best.coefs
                atp = self.prune_trees(at, intercept, coefs, index)
                
                if not self.silent:
                    print(f"mtrain {best.clf_mtrain:.4f}")
                    print(
                        f"  atp  {self.score(self._transformy(target, self.d.ytrain), at_predlab(atp, self.d.xtrain)):.4f}",
                        atp,
                    )
                    print(
                        f"  at   {self.score(self._transformy(target, self.d.ytrain), at_predlab(at, self.d.xtrain)):.4f}",
                        at,
                    )
                    print(f"mvalid {best.clf_mvalid:.4f}")
                    print(
                        f"  atp  {self.score(self._transformy(target, self.d.yvalid), at_predlab(atp, self.d.xvalid)):.4f}"
                    )
                    print(
                        f"  at   {self.score(self._transformy(target, self.d.yvalid), at_predlab(at, self.d.xvalid)):.4f}"
                    )

                self.at_singletarget[target] = atp
                bests.append(best)
            else:
                bests.append(None)
                # TODO: This is a quick fix, making sure that no REGR_MEAN trees are added to new_full_at, which is REGR.
                # This can happen when the tree is not updated, the learning rate (1/nb_trees) has to be pushed down first.
                new_at = at.copy()
                if at.get_type() == veritas.AddTreeType.REGR_MEAN:
                    nb_trees = len(new_at)
                    for t in new_at:
                        for leaf in t.get_leaf_ids():
                            t.set_leaf_value(leaf, t.get_leaf_value(leaf, 0)/nb_trees)
                atp = new_at
                print(
                    f"WARNING: weights of target {target} not updated,",
                    "combined model might fail",
                )

            new_full_at.add_trees(atp, target)
            
            # Explicit memory cleanup between targets to reduce memory pressure
            del xxtrain, yytrain, xxvalid, yyvalid, clf, alpha_search
            if 'best' in locals():
                del best
            import gc
            gc.collect()

        self.at = new_full_at

        # record
        #
        record = CompressRecord(
            level=level,
            at=self.at,
            mtrain=self.score(self.d.ytrain, at_predlab(self.at, self.d.xtrain)),
            mtest=self.score(self.d.ytest, at_predlab(self.at, self.d.xtest)),
            mvalid=self.score(self.d.yvalid, at_predlab(self.at, self.d.xvalid)),
            alphas=[b.alpha if b is not None else -1.0 for b in bests],
            clf_mtrain=[b.clf_mtrain if b is not None else np.nan for b in bests],
            clf_mvalid=[b.clf_mvalid if b is not None else np.nan for b in bests],
            tindex=tindex,
            ttransform=ttransform,
            tsearch=tsearch,
        )

        return record

    # def compress_level_fortarget(self, target, level, index):

    #    ttransform = time.time()
    #    xxtrain = self._transformx_fortarget(target, self.d.xtrain, index)
    #    yytrain = self._transformy_fortarget(target, self.d.ytrain)
    #    xxvalid = self._transformx_fortarget(target, self.d.xvalid, index)
    #    yyvalid = self._transformy_fortarget(target, self.d.yvalid)
    #    ttransform = time.time() - ttransform

    #    if not self.silent:
    #        print(
    #            f"Level {level}, target {target}, xxtrain.shape {xxtrain.shape},",
    #            "dense," if isinstance(xxtrain, np.ndarray) else "sparse,",
    #            f"transform time: {ttransform:.2f}s"
    #        )
    #    alpha_search = AlphaSearch(self, self.isworse)
    #    clf = self._get_regularized_lin_clf(xxtrain)

    #    tsearch = time.time()
    #    for alpha_record in alpha_search:
    #        self._update_lin_clf_alpha(clf, alpha_record.alpha)
    #        clf = self.fit_coefficients(clf, xxtrain, yytrain, xxvalid, yyvalid, alpha_record)

    #        if not self.silent:
    #            print_fit(alpha_record, alpha_search)

    #        #atp = self.prune_trees(clf.intercept_, clf.coef_, indexes)
    #        #print(atp)
    #        #print(clf.intercept_)
    #        #print(clf.coef_)

    #        #print(self._clf_decision_fun(clf, xxtrain))
    #        #print(atp.eval(self.d.xtrain))
    #        #print(self._clf_decision_fun(clf, xxtrain) - atp.eval(self.d.xtrain))
    #
    #        #raise RuntimeError("check")

    #    tsearch = time.time() - tsearch

    #    return alpha_search.get_best_record()

    def fit_coefficients(self, clf, xxtrain, yytrain, xxvalid, yyvalid, alpha_record):
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        fit_time = time.time()
        if self.no_convergence_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                clf.fit(xxtrain, yytrain)
        else:
            clf.fit(xxtrain, yytrain)

        # print(clf.n_iter_)
        fit_time = time.time() - fit_time
        # print(fit_time)

        num_params = xxtrain.shape[1]
        num_removed = np.sum(np.abs(clf.coef_) < 1e-5)
        frac_removed = num_removed / num_params

        yhat_train = clf.predict(xxtrain)
        yhat_valid = clf.predict(xxvalid)

        alpha_record.clf_mtrain = self.score(yytrain, yhat_train)
        alpha_record.clf_mvalid = self.score(yyvalid, yhat_valid)
        alpha_record.num_params = num_params
        alpha_record.num_removed = num_removed
        alpha_record.num_kept = num_params - num_removed
        alpha_record.frac_removed = frac_removed
        alpha_record.intercept = np.copy(clf.intercept_)
        alpha_record.coefs = np.copy(clf.coef_)
        alpha_record.fit_time = fit_time

        return clf

    # def _clf_decision_fun(self, clf, xx):
    #    if self.is_regression():
    #        return clf.predict(xx).reshape(-1, 1)
    #    else:
    #        ypred = clf.decision_function(xx)
    #        num_ex = xx.shape[0] // self.nlv
    #        ytargets = []
    #        for k in range(self.nlv):
    #            ytarget = ypred[k * num_ex : (k + 1) * num_ex].reshape(-1, 1)
    #            ytargets.append(ytarget)
    #        return np.hstack(ytargets)

    # def _clf_predict(self, clf, xx):
    #    df = self._clf_decision_fun(clf, xx)
    #    if self.is_regression():
    #        return df
    #    elif self.nlv == 1: # binary classification
    #        return df > 0.0
    #    else:
    #        return df.argmax(axis=1)

    def _get_regularized_lin_clf(self, xxtrain):
        self.seed += 1
        # print(self.seed)

        if self.is_regression():
            # Use memory-efficient solver settings for large matrices
            # Disable precompute for large matrices to save memory
            use_precompute = isinstance(xxtrain, np.ndarray) and xxtrain.shape[1] < 5000
            
            
            
            return Lasso(
                fit_intercept=self.fit_intercept,
                alpha=1.0,
                random_state=self.seed,
                max_iter=5000,
                tol=1e-5,
                warm_start=True,
                selection='random',
                copy_X=False,
                precompute=use_precompute,
            )
        else:
            # Use memory-efficient settings for large matrices
            
            
            
            return LogisticRegression(
                fit_intercept=self.fit_intercept,
                penalty="l1",
                C=1.0,
                solver="liblinear",
                max_iter=5000,
                tol=1e-5,
                n_jobs=1,
                random_state=self.seed,
                warm_start=True,
            )

    def _update_lin_clf_alpha(self, clf, alpha):
        if alpha <= 0.0:
            raise RuntimeError("alpha == 0.0?")

        if self.is_regression():
            assert isinstance(clf, Lasso)
            clf.alpha = 0.0001 * alpha
        else:
            assert clf.penalty == "l1"
            clf.C = 1.0 / alpha

    def _new_empty_addtree(self, num_leaf_values):
        if self.is_regression():
            at_type = veritas.AddTreeType.REGR
        else:
            at_type = veritas.AddTreeType.CLF_SOFTMAX
        return veritas.AddTree(num_leaf_values, at_type)

    def prune_trees(self, at, intercept, coefs, index):
        if self.is_regression():
            base_score = intercept if self.fit_intercept else 0.0
            coefs = coefs[:]
        else:
            base_score = intercept[0] if self.fit_intercept else 0.0
            coefs = coefs[0, :]
        atp = self._new_empty_addtree(1)
        atpp = atp.copy()
        offset = 0

        for m, t, (index0, index1) in zip(range(len(at)), at, index):
            # special case: if coef of full tree is 0.0, then drop the tree
            if t.root() in index1:
                offset = index1[t.root()]
                coef = coefs[offset]
                if abs(coef) < 1e-5:
                    continue
            self._copy_tree(t, atp.add_tree(), coefs, index1)

            tp = atp[len(atp) - 1]
            pruner = _TreeZeroLeafPruner(tp)
            if not pruner.is_root_zero():
                tpp = atpp.add_tree()
                pruner.prune(tp.root(), tpp, tpp.root())

        atp.set_base_score(0, base_score)
        atpp.set_base_score(0, base_score)
        return atpp

    # def prune_trees(self, intercept, coefs, indexes):
    #    if self.is_regression():
    #        coefs = coefs[:]
    #        at_type = veritas.AddTreeType.REGR
    #    else:
    #        coefs = coefs[0, :]
    #        at_type = veritas.AddTreeType.CLF_SOFTMAX

    #    if self.nlv == 1:
    #        index, num_cols = indexes[0]
    #        return self.prune_trees_fortarget(self.at, at_type, intercept, coefs, index)
    #    else:
    #        atp_full = veritas.AddTree(self.nlv, at_type)
    #        num_cols_offset = 0

    #        # Extract the pruned trees per target, and combine it in the single
    #        # multiclass/multitarget AddTree ensemble
    #        for k in range(self.nlv):
    #            index, num_cols = indexes[k]
    #            coefs_pertarget = coefs[num_cols_offset:num_cols_offset+num_cols]
    #            num_cols_offset += num_cols
    #            atp_pertarget = self.prune_trees_fortarget(
    #                self.at_singletarget[k], at_type, coefs_pertarget, index
    #            )

    #            atp_full.add_trees(atp_pertarget, k)
    #            atp_full.set_base_score(k, atp_pertarget.get_base_score(0))

    #        return atp_full

    # def prune_trees_fortarget(self, at, at_type, intercept, coefs, index):
    #    print(intercept)
    #    print(coefs)
    #    base_score = intercept[0]

    #    atp = veritas.AddTree(1, at_type)
    #    atpp = atp.copy()
    #    offset = 0

    #    for m, t, (index0, index1) in zip(range(len(at)), at, index):
    #        # special case: if coef of full tree is 0.0, then drop the tree
    #        if t.root() in index1:
    #            offset = index1[t.root()]
    #            coef = coefs[offset]
    #            if coef == 0.0:
    #                continue
    #        self._copy_tree(t, atp.add_tree(), coefs, index1)

    #        tp = atp[len(atp) - 1]
    #        pruner = _TreeZeroLeafPruner(tp)
    #        if not pruner.is_root_zero():
    #            tpp = atpp.add_tree()
    #            pruner.prune(tp.root(), tpp, tpp.root())

    #    atp.set_base_score(0, base_score)
    #    atpp.set_base_score(0, base_score)
    #    return atpp

    def _copy_tree(self, t, tc, coefs, index1):
        self._copy_subtree(t, t.root(), tc, tc.root(), coefs, index1)

    def _copy_subtree(self, t, n, tc, nc, coefs, index1):
        stack = [(n, nc, 1.0, 0.0)]
        while len(stack) > 0:
            n, nc, coef, bias = stack.pop()

            if n in index1:
                assert coef == 1.0
                assert bias == 0.0

                offset = index1[n]
                if t.is_root(n):
                    bias, coef = 0.0, coefs[offset]
                elif t.is_leaf(n):
                    bias, coef = coefs[offset], 0.0
                else:
                    bias, coef = coefs[offset : offset + 2]

                if abs(coef) <= 1e-5:  # skip the branch, just predict bias
                    # print(f"cutting off branch {n}, leaf value {bias:.3f}")
                    tc.set_leaf_value(nc, 0, bias)
                    continue

            if t.is_internal(n):
                s = t.get_split(n)
                tc.split(nc, s.feat_id, s.split_value)
                stack.append((t.right(n), tc.right(nc), coef, bias))
                stack.append((t.left(n), tc.left(nc), coef, bias))
            else:
                leaf_value = bias + coef * t.get_leaf_value(n, 0)
                tc.set_leaf_value(nc, 0, leaf_value)


class _TreeZeroLeafPruner:
    def __init__(self, t):
        self.t = t
        self.is_zero = np.zeros(t.num_nodes(), dtype=bool)
        self._can_prune(t.root())

    # Mark which subtrees all have leaf values equal, and can be pruned
    def _can_prune(self, n):
        t = self.t
        if t.is_leaf(n):
            is_zero = t.get_leaf_value(n, 0) == 0.0
        else:
            is_zero_right = self._can_prune(t.right(n))
            is_zero_left = self._can_prune(t.left(n))
            is_zero = is_zero_left and is_zero_right
        self.is_zero[n] = is_zero
        return is_zero

    def is_root_zero(self):
        return self.is_zero[self.t.root()]

    # Copy the tree, skipping the prunable nodes
    def prune(self, n, tc, nc):
        t = self.t
        if t.is_internal(n):
            left, right = t.left(n), t.right(n)
            if not (self.is_zero[left] and self.is_zero[right]):
                s = t.get_split(n)
                tc.split(nc, s.feat_id, s.split_value)
                self.prune(right, tc, tc.right(nc))
                self.prune(left, tc, tc.left(nc))
            else:
                tc.set_leaf_value(nc, 0, 0.0)
        else:
            lv = t.get_leaf_value(n, 0)
            tc.set_leaf_value(nc, 0, lv)