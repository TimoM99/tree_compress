import dis
import os
from turtle import pos
from unittest import result
os.environ['PRADA_DATA_DIR']='/cw/dtaijupiter/NoCsBack/dtai/timo/prada_data'
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import time
import random
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
import util
import veritas
from veritas import Solution
from numba import njit
from verification import run_verification_tasks

# def min_dist_to_solutions(example, intervals):
#     min_dist = float('inf')
#     dt = 0.0
#     # dist_per_feature = np.zeros(example.shape)
#     start = time.time()
#     for sol in intervals:
#         #TODO This is L_inf distance.
#         dist = dist_to_intervals_numba(example, sol[0], sol[1])
#         # closest = veritas.get_closest_example(sol, example, eps=0.0)
#         # dist = np.max(np.abs(closest - example))
#         if dist < min_dist:
#             min_dist = dist
#     dt += time.time() - start
#     return min_dist, dt

# @njit
# def dist_to_intervals_numba(x, ind, doms):
#     max_dist = 0
#     for i, idx in enumerate(ind):
#         if x[idx] < doms[i][0]:
#             max_dist = max(max_dist, doms[i][0] - x[idx])
#         elif x[idx] > doms[i][1]:
#             max_dist = max(max_dist, x[idx] - doms[i][1])
#     return max_dist

@njit
def min_dist_to_solutions(example, all_inds, all_doms, n_intervals):
    min_dist = 1e18
    for s in range(n_intervals):
        max_dist = 0.0
        inds = all_inds[s]
        doms = all_doms[s]
        for i in range(len(inds)):
            idx = inds[i]
            if idx == -1:
                break
            x = example[idx]
            low, high = doms[i, 0], doms[i, 1]
            if x < low:
                d = low - x
            elif x > high:
                d = x - high
            else:
                d = 0.0
            if d > max_dist:
                max_dist = d
        if max_dist < min_dist:
            min_dist = max_dist
    return min_dist


# def exact_emp_robustness(at, example, target_label):
#     from gurobipy import GRB

#     kan = veritas.KantchelianAttack(at, target_label, example)
#     kan.model.setParam(GRB.Param.TimeLimit, 10*60.0)
#     kan.model.setParam(GRB.Param.Threads, 1)
#     kan.optimize()
#     return kan.bounds[-1][0]

# TODO This is an alternative implementation of get_closest_example that does not work as well as the one in veritas.
# Parallellization does not help much because it requires more steps than the couple of if-statements in veritas.get_closest_example
# def get_closest_example_alt(solution_or_box, example, eps):

#     if isinstance(solution_or_box, Solution):
#         box = solution_or_box.box()
#     elif isinstance(solution_or_box, list):
#         if isinstance(solution_or_box[0], tuple):
#             box = {x[0]: x[1] for x in solution_or_box}
#         else:
#             box = {i: x for i, x in enumerate(solution_or_box)}
#     elif isinstance(solution_or_box, dict):
#         box = solution_or_box
#     else:
#         raise ValueError("invalid first argument")

#     lower = np.zeros(len(box))
#     upper = np.zeros(len(box))
#     indices = np.zeros(len(box))

#     for i, interval in enumerate(box.items()):
#         index, dom = interval
#         #TODO This is not the proper way to handle unbounded domains, but might work because of normalized data
#         # We do this so that we can know when an instance falls in the interval or outside.
#         lower[i] = max(dom.lo - eps, -10)
#         upper[i] = min(dom.hi + eps, 10)
#         indices[i] = index
#         indices = indices.astype(int)

#     dist_lower = np.abs(lower - example[indices])
#     dist_upper = np.abs(upper - example[indices])

#     closest = np.where(dist_lower < dist_upper, lower, upper)
#     closest = np.where(np.isclose(np.abs(dist_lower - dist_upper), np.abs(lower - upper)), closest, example[indices])

#     result = example.copy()
#     result[indices] = closest
#     return result



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

# config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
# config.stop_when_optimal = False
# config.max_memory = 16*1024*1024*1024
# search = config.get_search(at_orig)

# has_timed_out = False
# oom = False

# while True:
#     stop_reason = search.steps(1000)

#     if stop_reason == veritas.StopReason.NO_MORE_OPEN:
#         break

#     has_timed_out = search.time_since_start() >= 21600
#     oom = stop_reason == veritas.StopReason.OUT_OF_MEMORY
#     if has_timed_out or oom:
#         break

# out_of_resources = has_timed_out or oom

# print("Num solutions:", search.num_solutions())

# pos_solutions = []
# neg_solutions = []

# s = True
# i = 0
# while s:
#     try:
#         sol = search.get_solution(i)
#         pos_solutions.append(sol.box()) if sol.output > 0 else neg_solutions.append(sol.box())
        
#     except IndexError:
#         s = False
#     i += 1


# print(len(pos_solutions), len(neg_solutions))
print("Num leafs: ", at_orig.num_leafs())
print("Num leafs per tree:")
for t in at_orig:
    print(t.num_leaves(), end=' ')
nb_splits_per_feature = list(map(lambda y: len(y), at_orig.get_splits().values()))
print(at_orig.get_splits())
print('\nNb splits per feature', nb_splits_per_feature)
print('Bound on #solutions:', np.prod(np.array(nb_splits_per_feature) + 1))

# Linear scan over OC space
# # n = 500
# # count = 0

# # x = dtest.X
# # y = dtest.y

# # # print(dtest.X.loc[0, :].to_numpy())
# # start_time = time.time()
# # delta_tot = 0.0
# # dtt = 0.0

# # pos_solutions = [(list(sol.keys()), [(dom.lo, dom.hi) for dom in sol.values()]) for sol in pos_solutions]
# # max_k = max(len(sol[0]) for sol in pos_solutions)
# # n_pos = len(pos_solutions)

# # pos_inds = -np.ones((n_pos, max_k), dtype=np.int32)
# # pos_doms = np.zeros((n_pos, max_k, 2), dtype=np.float32)

# # for i, sol in enumerate(pos_solutions):
# #     k = len(sol[0])
# #     pos_inds[i, :k] = sol[0]
# #     pos_doms[i, :k] = sol[1]

# # neg_solutions = [(list(sol.keys()), [(dom.lo, dom.hi) for dom in sol.values()]) for sol in neg_solutions]
# # max_k = max(len(sol[0]) for sol in neg_solutions)
# # n_neg = len(neg_solutions)

# # neg_inds = -np.ones((n_neg, max_k), dtype=np.int32)
# # neg_doms = np.zeros((n_neg, max_k, 2), dtype=np.float32)



# # for i, sol in enumerate(neg_solutions):
# #     k = len(sol[0])
# #     neg_inds[i, :k] = sol[0]
# #     neg_doms[i, :k] = sol[1]

# # print('Transforming data types time: ', time.time() - start_time)
# # # print(pos_solutions[0])
# # for i in x.index:
# #     target_label = not (y.loc[i] > 0.0)
# #     example = x.loc[i, :].to_numpy()
# #     pred_label = at_orig.eval(example)[0, 0] > 0.0


# #     if pred_label != target_label:
# #         if target_label:
# #             # Find minimum distance to all positive solutions
# #             min_dist = min_dist_to_solutions(example, pos_inds, pos_doms, len(pos_solutions))

# #         else:
# #             # Find minimum distance to all negative solutions
# #             min_dist = min_dist_to_solutions(example, neg_inds, neg_doms, len(neg_solutions))
# #         # dtt += dt
# #         delta_tot += min_dist
# #         count += 1
# #         if count > n:
# #             break

# print("Avg delta linear scan:", delta_tot/count)
# print("Time in veritas.get_closest_example:", dtt)
# print('Time linear scan:', time.time() - start_time)

# # Exact verification MILP using Kantchelian et al.
# start_time = time.time()
# delta_tot = 0.0
# count = 0
# for i in x.index:
#     target_label = not (y.loc[i] > 0.0)
#     example = x.loc[i, :].to_numpy()
#     pred_label = at_orig.eval(example)[0, 0] > 0.0


#     if pred_label != target_label:
#         min_dist = exact_emp_robustness(at_orig, example, target_label)
#         delta_tot += min_dist
#         count += 1
#         if count > n:
#             break

# print("Avg delta exact:", delta_tot/count)
# print('Time exact:', time.time() - start_time)

results = run_verification_tasks(at_orig, dtest.X, dtest.y, timeout=21600, n=500)
print(results)