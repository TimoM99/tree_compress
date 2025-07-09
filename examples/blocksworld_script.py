import sys
import numpy as np
import prada
import veritas
import tree_compress
from sklearn.metrics import accuracy_score, balanced_accuracy_score


#dnames = ["Blocksworld_4_64_CA", "Blocksworld_6_64_CA", "Blocksworld_8_64_CA"]
#dnames = ["Blocksworld_4_32_CI", "Blocksworld_4_64_CA", "Blocksworld_6_32_CI"]
dname = "Mnist"
dname = "Blocksworld_4_32_CI"
param_dict = {
    "n_estimators": [20],
    "learning_rate": [0.5],
    "subsample": [0.5],
    "max_depth": [5],
    #"multi_strategy": "multi_output_tree"
}

def split_data(d):
    dtrain, dtest = d.split(0.66)
    dtest, dvalid = dtest.split(0.5)

    return dtrain, dvalid, dtest

# -----------------------------------------------------------------------------

best_models = {}
dd = prada.get_dataset(dname, silent=True, seed=12)
dd.load_dataset()
dd.robust_normalize()
dd.astype(np.float32)
dd.astype(np.float64)

dd = dd.to_argmax_multiclass()
d = dd.as_regression_problem()

dtrain, dvalid, dtest = split_data(d)

print("dtrain", dtrain.X.shape)
print("dtest", dtest.X.shape)
print("dvalid", dvalid.X.shape)

model_type = "xgb" # or "rf", "lgb"
model_class = d.get_model_class(model_type)

models = []
for i, params in enumerate(d.paramgrid(**param_dict)):
    clf, train_time = dtrain.train(model_class, params)

    mtrain = dtrain.metric(clf)
    mtest  = dtest.metric(clf)
    mvalid = dvalid.metric(clf)

    #              0  1    2       3      4       5
    models.append((d, clf, mtrain, mtest, mvalid, params))
    
    #print(f" - {i:4d} {dname:10s} train {mtrain*100:5.1f}%, valid {mvalid*100:5.1f}%, test {mtest*100:5.1f}%")

#best_mvalid = max(models, key=lambda m: m[4])
#for d, clf, mtrain, mtest, mvalid, params in models:
#    if abs(mvalid-best_mvalid[4]) < 0.001:
#        break
d, clf, mtrain, mtest, mvalid, params = max(models, key=lambda m: m[3])

best_models[d.name()] = (d, clf, mtrain, mtest, mvalid, params)
print(f"SELECTED {dname:10s} train {mtrain*100:5.1f}%, valid {mvalid*100:5.1f}%, test {mtest*100:5.1f}%")
print("params", params, d.X.shape, f"{d.num_classes} classes")
print()
    
del models

# -----------------------------------------------------------------------------


compressed_models = {}
for dname, (d, clf, mtrain, mtest, mvalid, params) in best_models.items():
    dtrain, dvalid, dtest = split_data(d)

    at = veritas.get_addtree(clf)
    nlv = at.num_leaf_values()

    at_compr = veritas.AddTree(nlv, veritas.AddTreeType.CLF_MEAN)
    at_singles = []
    
    for target in range(nlv):
        at_compr = at; break
        print()
        print(target, "===========================")
        at_single = at.make_singleclass(target)
        data = tree_compress.Data(
            dtrain.X.to_numpy(), dtrain.y.to_numpy()[:, target],
            dtest.X.to_numpy(), dtest.y.to_numpy()[:, target],
            dvalid.X.to_numpy(), dvalid.y.to_numpy()[:, target],
        )

        def my_single_metric(ytrue, ypred):
            return balanced_accuracy_score(ytrue > 0.0, ypred > 0.0)

        abserr = 0.01
        compressor = tree_compress.LassoCompress(
            data,
            at_single,
            metric=my_single_metric,
            isworse=lambda v, ref: ref-v > abserr,
            silent=False
        )
        #compressor.mtrain = mtrain
        #compressor.mtest = mtest
        #compressor.mvalid = mvalid
        compressor.no_convergence_warning = True
        at_single_compr = compressor.compress(max_rounds=1)

        at_compr.add_trees(at_single_compr, target)
        at_singles.append(at_single_compr)

    print()
    print("final ===========================")
    data = tree_compress.Data(
        dtrain.X.to_numpy(), dtrain.y.to_numpy(),
        dtest.X.to_numpy(), dtest.y.to_numpy(),
        dvalid.X.to_numpy(), dvalid.y.to_numpy()
    )

    def my_metric(ytrue, ypred):
        print(ytrue, "YTRUE")
        print(ypred, "YPRED")
        return accuracy_score(np.argmax(ytrue, axis=1), np.argmax(ypred, axis=1))

    abserr = 0.01
    compressor = tree_compress.LassoCompress(
        data,
        at_compr,
        metric=my_metric,
        isworse=lambda v, ref: ref-v > abserr,
        silent=False
    )
    print("is_regre", compressor.is_regression(), at_compr.get_type())
    compressor.mtrain = mtrain
    compressor.mtest = mtest
    compressor.mvalid = mvalid
    compressor.no_convergence_warning = True
    at_single_compr = compressor.compress(max_rounds=2)

    print(at.eval(dtest.X.iloc[0:5,:]).round(2))
    print(at_compr.eval(dtest.X.iloc[0:5,:]).round(2))
    print(dtest.y.iloc[0:5])
