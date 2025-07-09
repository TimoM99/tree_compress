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
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory(label):
    """Print current memory usage"""
    memory_mb = get_memory_usage()
    print(f"{label}: {memory_mb:.1f} MB")

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
        "n_estimators": 25,
        "max_depth": 6,
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


data = tree_compress.Data(
                    dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                    dtest.X.to_numpy(), dtest.y.to_numpy(),
                    dvalid.X.to_numpy(), dvalid.y.to_numpy())

print_memory("After data loading")
print(f"Data shapes: train={dtrain.X.shape}, test={dtest.X.shape}, valid={dvalid.X.shape}")
print(f"Model: {len(at_orig)} trees, {at_orig.num_leafs()} leafs, {at_orig.num_nodes()} nodes")



# # Compress xgb model using algorithm 1
# print_memory("Before LassoCompress")
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

# Compress xgb model using algorithm 2
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
at_refined_2 = compr.compress(max_rounds=2)
best_alpha_2 = compr.records[-1].alphas
compr_time_2 = time.time() - start_time
print_memory("After Compress compression")
print(f"Compress time: {compr_time_2:.2f}s")
for rec in compr.records:
    print(rec.alphas, rec.ntrees, rec.nleafs, rec.tindex, rec.tsearch, rec.ttransform)

print_memory("Final memory usage")

print(f"Compression times:, Compress={compr_time_2:.2f}s")
print(f"Trees: orig={at_orig.num_leafs()}, refined_2={at_refined_2.num_leafs()}")
print(f"Alphas: refined_2={best_alpha_2}")
print(f"Metrics: orig={dtest.metric(at_orig):.4f}, refined_2={dtest.metric(at_refined_2):.4f}")

print(f"Base scores: refined_2={at_refined_2.get_base_score(0)}")