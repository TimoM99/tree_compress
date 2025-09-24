import os
os.environ['PRADA_DATA_DIR']='/cw/dtaijupiter/NoCsBack/dtai/timo/prada_data'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import random
import util
import veritas
import tree_compress
import time
import psutil
import gc
from verification import run_verification_tasks
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory(label):
    """Print current memory usage"""
    memory_mb = get_memory_usage()
    print(f"{label}: {memory_mb:.1f} MB")

def feature_usage(at):
    original_features = set()
    for tree in at:
        def extract_features_recursive(node):
            if tree.is_leaf(node):
                return
            else:
                split = tree.get_split(node)
                original_features.add(split.feat_id)
                left_child = tree.left(node)
                right_child = tree.right(node)
                extract_features_recursive(left_child)
                extract_features_recursive(right_child)

        extract_features_recursive(tree.root())
    return original_features

print_memory("Start")


seed = 7
model_type = 'xgb'
dname = 'California'
regression = False  # Set to True for regression, False for classification
abserr = 0.005
fold = 1
silent = True
params = {
        "random_state": seed,
        "n_jobs": 1,
        "nthread": 1,
        "n_estimators": 10,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "tree_method": "hist",
    }

if regression == True:
    is_worse = lambda v, ref: ref*(1 + abserr) < v
    score = mean_squared_error
else:
    is_worse = lambda v, ref: ref - v > abserr
    score = balanced_accuracy_score

np.random.seed(seed)
random.seed(seed)

d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)
model_class = d.get_model_class(model_type)

# Fit XGB model
clf, _ = dtrain.train(model_class, params)
at_orig = veritas.get_addtree(clf, silent=silent)

def bound_oc_space(at):
    splits = at.get_splits()
    bound_1 = 1
    for _, f_values in splits.items():
        bound_1 *= (len(f_values) + 1)

    bound_2 = 1
    for t in at:
        bound_2 *= (t.num_leaves())
    return bound_1, bound_2
        

print(bound_oc_space(at_orig))

# print(len(feature_usage(at_orig)))


data = tree_compress.Data(
                    dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                    dtest.X.to_numpy(), dtest.y.to_numpy(),
                    dvalid.X.to_numpy(), dvalid.y.to_numpy())

print_memory("After data loading")
print(f"Data shapes: train={dtrain.X.shape}, test={dtest.X.shape}, valid={dvalid.X.shape}")
print(f"Model: {len(at_orig)} trees, {at_orig.num_leafs()} leafs, {at_orig.num_nodes()} nodes")



# # # Compress xgb model using OC compress
# print_memory("Before OC Compress")
# start_time = time.time()
# compr = tree_compress.LassoCompress( # For binary classification, use LassoCompress.
#             data,
#             at_orig,
#             metric=score,
#             isworse=is_worse,
#             linclf_type='LogisticRegression',
#             seed=5823,
#             silent=silent
#         )

# print_memory("After LassoCompress init")

# compr.no_convergence_warning = True
# at_refined_1 = compr.compress(max_rounds=2)
# best_alpha_1 = compr.records[-1].alpha
# compr_time_1 = time.time() - start_time
# print_memory("After LassoCompress compression")
# print(f"LassoCompress time: {compr_time_1:.2f}s")
# for rec in compr.records:
#     print(rec.alpha, rec.ntrees, rec.nleafs, rec.tmapping, rec.tsearch, rec.ttransform)

# # Force garbage collection to free memory
# del compr, at_refined_1
# gc.collect()
# print_memory("After LassoCompress cleanup")

# Compress xgb model using OC Compress (Observable Coverage based)
print_memory("Before OC Compress")
start_time = time.time()
compr_oc = tree_compress.freeze_compress_pytorch.Compress(
            data,
            at_orig,
            score=score,
            isworse=is_worse,
            seed=5823,
            silent=True,
            frozen_pct=0.2,  # Percentage of features to freeze at every level
        )

print_memory("After OC Compress init")
compr_oc.no_convergence_warning = True
at_refined_oc = compr_oc.compress(max_rounds=2)
# at_refined_oc = at_orig
compr_time_oc = time.time() - start_time
print_memory("After OC Compress compression")
print(f"OC Compress time: {compr_time_oc:.2f}s")
for rec in compr_oc.records:
    print(rec.alphas, rec.ntrees, rec.nleafs, rec.tindex, rec.tsearch, rec.ttransform)

print_memory("After OC Compress cleanup")

# Compress xgb model using LOP
np.random.seed(seed)
random.seed(seed)

print_memory("Before Compress")
start_time = time.time()
compr = tree_compress.Compress(
            data,
            at_orig,
            score=score,
            isworse=is_worse,
            seed=5823,
            silent=True
        )

print_memory("After Compress init")
compr.no_convergence_warning = True
at_refined_lop = compr.compress(max_rounds=2)
compr_time_lop = time.time() - start_time
print_memory("After Compress compression")
print(f"Compress time: {compr_time_lop:.2f}s")
for rec in compr.records:
    print(rec.alphas, rec.ntrees, rec.nleafs, rec.tindex, rec.tsearch, rec.ttransform)

print_memory("Final memory usage")

verification_results_orig = run_verification_tasks(at_orig, dtest.X, dtest.y, timeout=1800, n=500)
verification_results_oc = run_verification_tasks(at_refined_oc, dtest.X, dtest.y, timeout=1800, n=500)
verification_results_lop = run_verification_tasks(at_refined_lop, dtest.X, dtest.y, timeout=1800, n=500)

print(f"Compression times: OC Compress={compr_time_oc:.2f}s, Compress={compr_time_lop:.2f}s")
print(f"Leafs: orig={at_orig.num_leafs()}, refined_OC={at_refined_oc.num_leafs()}, refined_Lop={at_refined_lop.num_leafs()}")
print(f"Trees: orig={len(at_orig)}, refined_OC={len(at_refined_oc)}, refined_Lop={len(at_refined_lop)}")
print(f"Test scores: orig={score(dtest.y, at_orig.predict(dtest.X)>0.5)}, refined_OC={score(dtest.y, at_refined_oc.predict(dtest.X)>0.5)}, refined_Lop={score(dtest.y, at_refined_lop.predict(dtest.X)>0.5)}")
print(f"Valid scores: orig={score(dvalid.y, at_orig.predict(dvalid.X)>0.5)}, refined_OC={score(dvalid.y, at_refined_oc.predict(dvalid.X)>0.5)}, refined_Lop={score(dvalid.y, at_refined_lop.predict(dvalid.X)>0.5)}")
print(f"Base scores: orig={at_orig.get_base_score(0)}, refined_OC={at_refined_oc.get_base_score(0)}, refined_Lop={at_refined_lop.get_base_score(0)}")



print(f"Features used: orig={len(feature_usage(at_orig))}, refined_OC={len(feature_usage(at_refined_oc))}, refined_Lop={len(feature_usage(at_refined_lop))}")
print("Verification results (orig, OC, Lop):")
for key in verification_results_orig.keys():
    res_orig = verification_results_orig[key]
    res_oc = verification_results_oc.get(key, None)
    res_lop = verification_results_lop.get(key, None)
    print(f"  {key}: orig={res_orig}, OC={res_oc}, Lop={res_lop}")



