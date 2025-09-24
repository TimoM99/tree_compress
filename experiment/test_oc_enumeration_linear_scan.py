import os
os.environ['PRADA_DATA_DIR']='/cw/dtaijupiter/NoCsBack/dtai/timo/prada_data'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import random
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
import util
import veritas

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
        "n_estimators": 5,
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

config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
config.stop_when_optimal = False
config.max_memory = 16*1024*1024*1024
search = config.get_search(at_orig)

has_timed_out = False
oom = False

while True:
    stop_reason = search.steps(1000)

    if stop_reason == veritas.StopReason.NO_MORE_OPEN:
        break

    has_timed_out = search.time_since_start() >= 21600
    oom = stop_reason == veritas.StopReason.OUT_OF_MEMORY
    if has_timed_out or oom:
        break

out_of_resources = has_timed_out or oom

print("Num solutions:", search.num_solutions())
solutions = []

s = True
i = 0
while s:
    try:
        solutions.append(search.get_solution(i))
    except IndexError:
        s = False
    i += 1
print(len(solutions))