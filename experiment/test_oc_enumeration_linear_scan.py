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


def min_dist_to_solutions(example, solutions):
    min_dist = float('inf')
    for sol in solutions:
        closest = veritas.get_closest_example(sol, example, eps=0.0)
        dist = np.max(np.abs(example - closest))
        if dist < min_dist:
            min_dist = dist
    return min_dist

def exact_emp_robustness(at, example, target_label, max_delta):
    from gurobipy import GRB

    # box = [veritas.Interval(x-max_delta, x+max_delta) for x in example]
    # at_pruned = at.prune(box)
    kan = veritas.KantchelianAttack(at, target_label, example)
    kan.model.setParam(GRB.Param.TimeLimit, 10*60.0)
    kan.model.setParam(GRB.Param.Threads, 1)
    kan.optimize()
    return kan.bounds[-1][0]
    # try:
    #     return min(max_delta, kan.bounds[-1][0])
    # except IndexError:
    #     return max_delta


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
print(dtrain.X)
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
pos_solutions = []
neg_solutions = []

s = True
i = 0
while s:
    try:
        sol = search.get_solution(i)
        pos_solutions.append(sol) if sol.output > 0 else neg_solutions.append(sol)
    except IndexError:
        s = False
    i += 1

print(len(pos_solutions), len(neg_solutions))

n = 500
count = 0

x = dtest.X
y = dtest.y

delta_tot = 0.0
for i in x.index:
    target_label = not (y.loc[i] > 0.0)
    example = x.loc[i, :].to_numpy()
    pred_label = at_orig.eval(example)[0, 0] > 0.0


    if pred_label != target_label:
        if target_label:
            # Find minimum distance to all positive solutions
            min_dist = min_dist_to_solutions(example, pos_solutions)
        else:
            # Find minimum distance to all negative solutions
            min_dist = min_dist_to_solutions(example, neg_solutions)
        delta_tot += min_dist
        count += 1
        if count > n:
            break
        # print(min_dist, exact_emp_robustness(at_orig, example, target_label, max_delta=1.0))
        # assert np.isclose(min_dist, exact_emp_robustness(at_orig, example, target_label, max_delta=1.0), atol=1e-4)

print("Avg delta:", delta_tot/count)
