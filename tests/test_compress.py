import os
import time
import numpy as np
import unittest
import tree_compress
import veritas

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error

BPATH = os.path.dirname(__file__)


class TestCompress(unittest.TestCase):
    def test_binary_rf(self):
        self.do_binary(RandomForestClassifier)

    def test_binary_gbdt(self):
        self.do_binary(GradientBoostingClassifier, learning_rate=0.8)

    def do_binary(self, clazz, **kwargs):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        yreal = np.array([img[x, y] for x, y in X])
        X = X.astype(veritas.FloatT)

        ymedian = np.median(yreal)
        yclf = yreal > ymedian

        xtrain, xtest, ytrain, ytest = train_test_split(
            X, yclf, test_size=0.40, random_state=42
        )
        xvalid, xtest, yvalid, ytest = train_test_split(xtest, ytest, test_size=0.5)

        clf = clazz(
            n_estimators=100,
            max_depth=6,
            **kwargs
        )
        clf.fit(xtrain, ytrain)

        yhat1 = clf.predict(X)

        at = veritas.get_addtree(clf)
        veritas.test_conversion(at, X[:10,:], clf.predict_proba(X[:10,:])[:,1])
        yhat2 = at.predict(X) > 0.5

        data = tree_compress.Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
        compressor = tree_compress.Compress(data, at)
        relerr = 0.01
        at_pruned = compressor.compress(relerr=relerr, max_rounds=1)
        yhat3 = at_pruned.predict(X) > 0.5

        before = compressor.records[0]
        after = compressor.records[-1]

        self.assertGreater(before.ntrees, after.ntrees)
        self.assertGreater(before.nnodes, after.nnodes)
        self.assertGreater(before.nleafs, after.nleafs)
        self.assertGreater(before.nnz_leafs, after.nnz_leafs)
        self.assertGreaterEqual(after.mvalid+relerr, before.mvalid)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(yclf.reshape((100, 100)))
            ax[0, 0].set_title("ground truth")
            ax[0, 1].imshow(yhat1.reshape((100, 100)))
            ax[0, 1].set_title("clf pred")
            ax[1, 0].imshow(yhat2.reshape((100, 100)))
            ax[1, 0].set_title("at")
            ax[1, 1].imshow(yhat3.reshape((100, 100)))
            ax[1, 1].set_title("pruned")
            plt.tight_layout()
            plt.show()
        except ModuleNotFoundError:
            pass

    def test_regression_rf(self):
        self.do_regression(RandomForestRegressor)

    def test_regression_gbdt(self):
        self.do_regression(GradientBoostingRegressor, learning_rate=0.8)

    def do_regression(self, clazz, **kwargs):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        y = y.astype(veritas.FloatT) / float(np.max(y))
        X = X.astype(veritas.FloatT)

        def mymetric(ytrue, ypred):
            return root_mean_squared_error(ytrue, ypred)

        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.40, random_state=42
        )
        xvalid, xtest, yvalid, ytest = train_test_split(xtest, ytest, test_size=0.5)

        clf = clazz(
            n_estimators=40,
            max_depth=4,
            **kwargs
        )
        clf.fit(xtrain, ytrain)

        yhat1 = clf.predict(X)

        at_orig = veritas.get_addtree(clf)
        veritas.test_conversion(at_orig, X[:10,:], clf.predict(X[:10,:]))
        yhat2 = at_orig.predict(X)

        abserr = np.mean(np.abs(y - yhat2)) / 20.0
        print("absolute error:", abserr)
        data = tree_compress.Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
        compr = tree_compress.LassoCompress(
            data,
            at_orig,
            metric=mymetric,
            isworse=lambda v, ref: v-ref > abserr,
            linclf_type="Lasso",
            seed=123,
            silent=False
        )
        compr_time = time.time()
        at_compr = compr.compress(max_rounds=1)
        compr_time = time.time() - compr_time
        yhat3 = at_compr.predict(X)

        before = compr.records[0]
        after = compr.records[-1]

        self.assertGreaterEqual(before.ntrees, after.ntrees)
        self.assertGreater(before.nnodes, after.nnodes)
        self.assertGreater(before.nleafs, after.nleafs)
        self.assertGreater(before.nnzleafs, after.nnzleafs)
        #self.assertGreaterEqual(-(1+relerr)*after.mvalid, -before.mvalid)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
            ax[0, 0].imshow(y.reshape((100, 100)))
            ax[0, 0].set_title("ground truth")
            ax[0, 1].imshow(yhat1.reshape((100, 100)))
            ax[0, 1].set_title("clf pred")
            ax[1, 0].imshow(yhat2.reshape((100, 100)))
            ax[1, 0].set_title("at")
            ax[1, 1].imshow(yhat3.reshape((100, 100)))
            ax[1, 1].set_title("pruned")
            plt.tight_layout()
            plt.show()
        except ModuleNotFoundError:
            pass

    def test_multiclass_rf(self):
        self.do_multiclass(RandomForestClassifier)

    def test_multiclass_gbdt(self):
        self.do_multiclass(GradientBoostingClassifier)

    def do_multiclass(self, clazz, **kwargs):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        yreal = np.array([img[x, y] for x, y in X])

        quantiles = np.quantile(yreal, [0.33, 0.67])
        y = np.digitize(yreal, quantiles)
        y = y.astype(int)
        X = X.astype(veritas.FloatT)

        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.40, random_state=37
        )
        xvalid, xtest, yvalid, ytest = train_test_split(xtest, ytest, test_size=0.5, random_state=73)

        clf = clazz(
            random_state=37,
            n_estimators=20,
            max_depth=5,
            **kwargs
        )
        clf.fit(xtrain, ytrain)

        yhat1 = clf.predict(X)


        at = veritas.get_addtree(clf)
        veritas.test_conversion(at, X[:10,:], clf.predict_proba(X[:10,:]))
        yhat2 = np.argmax(at.predict(X), axis=1)

        abserr = 0.01
        data = tree_compress.Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
        compressor = tree_compress.LassoCompress(
            data,
            at,
            metric=accuracy_score,
            isworse=lambda v, ref: ref-v > abserr,
            silent=False
        )
        compressor.no_convergence_warning = True
        at_pruned = compressor.compress(max_rounds=2)
        yhat3 = np.argmax(at_pruned.predict(X), axis=1)
        before = compressor.records[0]
        after = compressor.records[-1]

        #self.assertGreaterEqual(before.ntrees, after.ntrees)
        #self.assertGreater(before.nnodes, after.nnodes)
        #self.assertGreater(before.nleafs, after.nleafs)
        #self.assertGreater(before.nnz_leafs, after.nnz_leafs)
        #self.assertGreaterEqual((1+relerr)*after.mvalid, before.mvalid)

        for t in at_pruned:
            print(t)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 3)
            ax[0, 0].imshow(y.reshape((100, 100)), interpolation="none")
            ax[0, 0].set_title("ground truth")
            ax[0, 1].imshow(yhat1.reshape((100, 100)), interpolation="none")
            ax[0, 1].set_title("clf pred")
            ax[1, 0].imshow(yhat2.reshape((100, 100)), interpolation="none")
            ax[1, 0].set_title("at (check)")
            ax[1, 1].imshow(yhat3.reshape((100, 100)), interpolation="none")
            ax[1, 1].set_title("compressed")
            ax[0, 2].imshow((yhat2 == y).reshape((100, 100)), cmap="Reds", interpolation="none")
            ax[0, 2].set_title("errors clf")
            ax[1, 2].imshow((yhat3 == y).reshape((100, 100)), cmap="Reds", interpolation="none")
            ax[1, 2].set_title("errors clf")

            fig, ax = plt.subplots(3, 3)
            for k in range(3):
                ax[k, 0].set_title(f"ground target {k}")
                ax[k, 0].imshow((y==k).reshape((100, 100)), interpolation="none")
                ax[k, 1].set_title(f"uncompr. target {k}")
                ax[k, 1].imshow(at.eval(X)[:,k].reshape((100, 100)), vmin=-1, vmax=1, interpolation="none")
                ax[k, 2].set_title(f"compr. target {k}")
                ax[k, 2].imshow(at_pruned.eval(X)[:,k].reshape((100, 100)), vmin=-1, vmax=1, interpolation="none")

            plt.tight_layout()
            plt.show()
        except ModuleNotFoundError:
            pass

if __name__ == "__main__":
    unittest.main()
