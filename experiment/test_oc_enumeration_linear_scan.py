import gc
from math import e
import os
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

import resource
from multiprocessing import Process, Pipe
import traceback

def set_memory_limit(byte: int):
    limit = byte
    # limit virtual memory (address space)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


# ------------------------------
# Universal runner for any function
# ------------------------------
def _worker(conn, func, args, kwargs, mem_limit):
    try:
        if mem_limit is not None:
            set_memory_limit(mem_limit)

        result = func(*args, **kwargs)
        conn.send(("ok", result))

    except MemoryError as e:
        conn.send(("error", "MemoryError: " + str(e)))

    except Exception as e:
        # Include traceback for debugging
        tb = traceback.format_exc()
        conn.send(("error", f"{e}\n{tb}"))


# ------------------------------
# Public API
# ------------------------------
def run_with_timeout_and_memory(func, *args, timeout=None, mem_limit=None, **kwargs):
    parent, child = Pipe()
    p = Process(target=_worker, args=(child, func, args, kwargs, mem_limit))
    p.start()

    start_time = time.time()
    while True:
        if parent.poll(0.1):  # check every 0.1 sec if worker sent data
            status, payload = parent.recv()
            p.join()
            if status == "ok":
                return payload
            if "MemoryError" in payload:
                raise MemoryError(payload)
            raise RuntimeError("Worker error:\n" + payload)
        
        if timeout is not None and (time.time() - start_time) > timeout:
            p.kill()
            p.join()
            raise TimeoutError(f"Function did not finish within {timeout} seconds")




def count_ocs(at, timeout):
    config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
    config.stop_when_optimal = False
    config.max_memory = 32*1024*1024*1024
    search = config.get_search(at)

    has_timed_out = False
    oom = False

    while True:
        stop_reason = search.steps(1000)
        if stop_reason == veritas.StopReason.NO_MORE_OPEN:
            break

        has_timed_out = search.time_since_start() >= timeout
        oom = stop_reason == veritas.StopReason.OUT_OF_MEMORY
        if has_timed_out or oom:
            break

    out_of_resources = has_timed_out or oom
    return search.num_solutions(), search.time_since_start(), out_of_resources

def find_ocs(at, timeout, memory_limit):
    config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
    config.stop_when_optimal = False
    config.max_memory = memory_limit
    search = config.get_search(at)

    has_timed_out = False
    oom = False

    while True:
        stop_reason = search.steps(1000)
        # print(search.get_used_memory())
        if stop_reason == veritas.StopReason.NO_MORE_OPEN:
            break

        has_timed_out = search.time_since_start() >= timeout
        oom = stop_reason == veritas.StopReason.OUT_OF_MEMORY
        if has_timed_out or oom:
            break

    out_of_resources = has_timed_out or oom
    time_taken = search.time_since_start()
    memory_used = search.get_used_memory()

    s = True
    i = 0
    pos_count = 0
    max_box_size = 0
    while s:
        try:
            sol = search.get_solution(i)
        except IndexError:
            break
        
        pos_count += 1 if sol.output > 0 else 0
        max_box_size = max(max_box_size, len(sol.box().keys()))
        i += 1

    num_solutions = search.num_solutions()
    # num_features = len(at.get_splits())

    pos_inds = -np.ones((pos_count, max_box_size), dtype=np.int32)
    neg_inds = -np.ones((num_solutions - pos_count, max_box_size), dtype=np.int32)
    pos_doms = np.zeros((pos_count, max_box_size, 2), dtype=np.float32)
    neg_doms = np.zeros((num_solutions - pos_count, max_box_size, 2), dtype=np.float32)

    p = 0
    n = 0
    while s:
        try:
            sol = search.get_solution(p+n)
        except IndexError:
            break
            # pos_solutions.append(sol.box()) if sol.output > 0 else neg_solutions.append(sol.box())
        label = 1 if sol.output > 0 else 0

        sol = sol.box()
        k = len(sol.keys())

        if label == 1:
            pos_inds[p, :k] = list(sol.keys())
            pos_doms[p, :k, 0] = [dom.lo for dom in sol.values()]
            pos_doms[p, :k, 1] = [dom.hi for dom in sol.values()]
            p += 1

        else:
            neg_inds[n, :k] = list(sol.keys())
            neg_doms[n, :k, 0] = [dom.lo for dom in sol.values()]
            neg_doms[n, :k, 1] = [dom.hi for dom in sol.values()]
            n += 1
        

    inds = {'positive': pos_inds, 'negative': neg_inds}
    doms = {'positive': pos_doms, 'negative': neg_doms}
    return inds, doms, out_of_resources, time_taken, memory_used

def find_ocs_alternative(at, timeout, memory_limit):
    start_time = time.time()
    out_of_resources = False
    memory_limit_reached = False

    doms = {'positive': np.full((0,0,2), 0), 'negative': np.full((0,0,2), 0)}
    inds = {'positive': np.full((0,0), 0), 'negative': np.full((0,0), 0)}
    try:
        inds, doms, preds = run_with_timeout_and_memory(
            enumerate_ocs, at, memory_limit, timeout=timeout, mem_limit=memory_limit
        )
    except (TimeoutError, MemoryError) as e:
        out_of_resources = True
        memory_limit_reached = isinstance(e, MemoryError)

    
    time_taken = time.time() - start_time

    return inds, doms, out_of_resources, time_taken, memory_limit_reached

def enumerate_ocs(at, max_mem):
    doms = None
    inds = None
    preds = None

    for t in at:
        leaf_ids = t.get_leaf_ids()

        leaf_boxes = [t.compute_box(lid) for lid in leaf_ids]
        leaf_preds = np.array([t.get_leaf_value(lid, 0) for lid in leaf_ids], dtype=np.float32)
        
        leaf_doms = np.zeros((len(leaf_boxes), max(len(box) for box in leaf_boxes), 2), dtype=np.float32)
        leaf_inds = -np.ones((len(leaf_boxes), max(len(box) for box in leaf_boxes)), dtype=np.int32)
        for k, box in enumerate(leaf_boxes):
            leaf_doms[k, :len(box.keys())] = [[dom.lo, dom.hi] for dom in box.values()]
            leaf_inds[k, :len(box.keys())] = list(box.keys())

        if doms is None:
            doms = leaf_doms
            inds = leaf_inds
            preds = leaf_preds + at.get_base_score(0)

        else:
            n, d = doms.shape[0], doms.shape[1]
            slice_size = int(max_mem * 0.3 / (d * 4 * 2 + d * 4 * 1 + 4))  # 30% of max memory for doms, inds, preds
            n_slices = int(np.ceil(n / slice_size))

            counts, max_size = [], 0
            for i in range(n_slices):
                count_slice, max_size_slice = cross_product_prepare(doms[i*slice_size : min((i+1)*slice_size, n)], inds[i*slice_size : min((i+1)*slice_size, n)], leaf_doms, leaf_inds)
                counts.append(count_slice)
                max_size = max(max_size, max_size_slice)

            doms_result = np.memmap('/cw/dtailocal/timo/OCs/temp_doms_{}.dat'.format(time.time()), shape=(np.sum(counts), max_size, 2), dtype=np.float32, mode='w+')
            inds_result = np.memmap('/cw/dtailocal/timo/OCs/temp_inds_{}.dat'.format(time.time()), shape=(np.sum(counts), max_size), dtype=np.int32, mode='w+')
            preds_result = np.memmap('/cw/dtailocal/timo/OCs/temp_preds_{}.dat'.format(time.time()), shape=(np.sum(counts),), dtype=np.float32, mode='w+')

            for i in range(n_slices):
                start_idx = np.sum(counts[:i], dtype=np.int32)
                end_idx = start_idx + counts[i]

                inds_result[start_idx:end_idx] = -1
                cross_product(doms[i*slice_size : min((i+1)*slice_size, n)], inds[i*slice_size : min((i+1)*slice_size, n)], preds[i*slice_size : min((i+1)*slice_size, n)], leaf_doms, leaf_inds, leaf_preds,
                              doms_result[start_idx:end_idx], inds_result[start_idx:end_idx], preds_result[start_idx:end_idx])

            doms_result.flush()
            inds_result.flush()
            preds_result.flush()

            try:
                os.remove(doms.filename)
                os.remove(inds.filename)
                os.remove(preds.filename)
            except AttributeError as e:
                pass

            doms = np.memmap(doms_result.filename, mode='r', shape=doms_result.shape, dtype=doms_result.dtype)
            inds = np.memmap(inds_result.filename, mode='r', shape=inds_result.shape, dtype=inds_result.dtype)
            preds = np.memmap(preds_result.filename, mode='r', shape=preds_result.shape, dtype=preds_result.dtype)

    return inds, doms, preds

@njit 
def cross_product_prepare(doms, inds, leaf_doms, leaf_inds):
    count = 0
    max_size = 0
    for i in range(doms.shape[0]):
        for j in range(leaf_doms.shape[0]):
            dom = doms[i]
            ind = inds[i]
            leaf_dom = leaf_doms[j]
            leaf_ind = leaf_inds[j]
            compatible, size = is_compatible(dom, ind, leaf_dom, leaf_ind)
            if compatible:
                count += 1
                max_size = max(max_size, size)

    return count, max_size

@njit
def cross_product(doms, inds, preds, leaf_doms, leaf_inds, leaf_preds, doms_result, inds_result, preds_result):
    k = 0
    for i in range(doms.shape[0]):
        for j in range(leaf_doms.shape[0]):
            dom = doms[i]
            ind = inds[i]
            pred = preds[i]
            leaf_dom = leaf_doms[j]
            leaf_ind = leaf_inds[j]
            leaf_pred = leaf_preds[j]
            compatible, _ = is_compatible(dom, ind, leaf_dom, leaf_ind)
            if compatible:
                merge_doms(dom, ind, leaf_dom, leaf_ind, doms_result[k], inds_result[k])
                preds_result[k] = pred + leaf_pred
                k += 1



# @njit
# def cross_product(doms, inds, preds, leaf_doms, leaf_inds, leaf_preds):
#     # print('inside')
#     count = 0
#     max_size = 0
#     for i in range(doms.shape[0]):
#         for j in range(leaf_doms.shape[0]):
#             dom = doms[i]
#             ind = inds[i]
#             leaf_dom = leaf_doms[j]
#             leaf_ind = leaf_inds[j]
#             compatible, size = is_compatible(dom, ind, leaf_dom, leaf_ind)
#             if compatible:
#                 count += 1
#                 max_size = max(max_size, size)
#     # print('WE GET HERE')
#     doms_result = np.zeros((count, max_size, 2), dtype=np.float32)
#     print(doms_result.shape)
#     inds_result = -np.ones((count, max_size), dtype=np.int32)
#     preds_result = np.zeros((count,), dtype=np.float32)
    
#     k = 0
#     for i in range(doms.shape[0]):
#         for j in range(leaf_doms.shape[0]):
#             dom = doms[i]
#             ind = inds[i]
#             pred = preds[i]
#             leaf_dom = leaf_doms[j]
#             leaf_ind = leaf_inds[j]
#             leaf_pred = leaf_preds[j]
#             compatible, _ = is_compatible(dom, ind, leaf_dom, leaf_ind)
#             if compatible:
#                 # Merge dom and leaf_dom
#                 # print(ind, leaf_ind)
#                 # print(doms_result[k].shape, inds_result[k].shape)
#                 # print('WE GET HERE')
#                 merge_doms(dom, ind, leaf_dom, leaf_ind, doms_result[k], inds_result[k])
#                 # print('after merge')
#                 preds_result[k] = pred + leaf_pred
#                 # print('Merged box: ', doms_result[k][:box_size], inds_result[k][:box_size])
#                 k += 1
    
#     return doms_result, inds_result, preds_result

@njit
def is_compatible(dom, ind, leaf_dom, leaf_ind):
    i = 0
    j = 0
    box_size = np.count_nonzero(ind != -1) + np.count_nonzero(leaf_ind != -1)
    while i < len(leaf_ind) and j < len(ind):
        l_idx = leaf_ind[i]
        idx = ind[j]
        if l_idx == -1 or idx == -1:
            break
        if l_idx < idx:
            i += 1
        elif l_idx > idx:
            j += 1
        else:
            low, high = leaf_dom[i, 0], leaf_dom[i, 1]
            d_low, d_high = dom[j, 0], dom[j, 1]
            if min(high, d_high) <= max(low, d_low):
                return False, None
            box_size -= 1
            i += 1
            j += 1
    return True, box_size

@njit
def merge_doms(dom, ind, leaf_dom, leaf_ind, dom_result, ind_result):
    # print('merging')
    i = 0
    j = 0
    l = 0
    while i < len(leaf_ind) and j < len(ind):
        # print(i, j)
        l_idx = leaf_ind[i]
        idx = ind[j]
        if l_idx == -1 and idx == -1:
            break
        if (l_idx < idx and l_idx != -1) or idx == -1:
            dom_result[l] = [leaf_dom[i, 0], leaf_dom[i, 1]]
            ind_result[l] = l_idx
            l += 1
            i += 1

        elif (l_idx > idx and idx != -1) or l_idx == -1:
            dom_result[l] = [dom[j, 0], dom[j, 1]]
            ind_result[l] = idx
            l += 1
            j += 1
        
        else:
            low, high = leaf_dom[i, 0], leaf_dom[i, 1]
            d_low, d_high = dom[j, 0], dom[j, 1]
            dom_result[l] = [max(low, d_low), min(high, d_high)]
            ind_result[l] = idx
            i += 1
            j += 1
            l += 1
    
    # print('step 1')
    # print(i, leaf_ind, j, ind, l)
    # print(dom_result.shape)
    
    if i < len(leaf_ind) and leaf_ind[i] != -1:
        nb_filled_leaf_ind = np.count_nonzero(leaf_ind != -1)

        dom_result[l:l + (nb_filled_leaf_ind - i), 0] = leaf_dom[i:nb_filled_leaf_ind, 0]
        dom_result[l:l + (nb_filled_leaf_ind - i), 1] = leaf_dom[i:nb_filled_leaf_ind, 1]
        ind_result[l:l + (nb_filled_leaf_ind - i)] = leaf_ind[i:nb_filled_leaf_ind]

    if j < len(ind) and ind[j] != -1:
        nb_filled_ind = np.count_nonzero(ind != -1)

        dom_result[l:l + (nb_filled_ind - j), 0] = dom[j:nb_filled_ind, 0]
        dom_result[l:l + (nb_filled_ind - j), 1] = dom[j:nb_filled_ind, 1]
        ind_result[l:l + (nb_filled_ind - j)] = ind[j:nb_filled_ind]

    # print('step 2')
    

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


seed = 7
model_type = 'xgb'
dname = 'Adult'
regression = False  # Set to True for regression, False for classification
abserr = 0.005
fold = 1
silent = True
params = {
        "random_state": seed,
        "n_jobs": 1,
        "nthread": 1,
        "n_estimators": 15,
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
    
start_time = time.time()
inds, doms, out_of_resources, time_taken, memory_used = find_ocs(at_orig, timeout=21600, memory_limit=16*1024*1024*1024)
time_taken = time.time() - start_time

print("Time taken for old enumeration:", time_taken)
# print("inds positive:", inds['positive'])
print("Number of solutions:", inds['positive'].shape[0], inds['negative'].shape[0])

# start_time = time.time()
# inds_alt, doms_alt, out_of_resources, time_taken, memory_used = find_ocs_alternative(at_orig, timeout=21600, memory_limit=16*1024*1024*1024)
# time_taken = time.time() - start_time

start_time = time.time()
inds_alt, doms_alt, out_of_resources, time_taken, memory_used = find_ocs_alternative(at_orig, timeout=86400, memory_limit=16*1024*1024*1024)
# inds_alt, doms_alt, out_of_resources, time_taken, memory_used = find_ocs_alternative(at_orig)
time_taken = time.time() - start_time
print("Time taken for alternative enumeration:", time_taken)
# print("inds positive alternative:", inds_alt)
# print("Domains positive alternative:", doms_alt)
# print("Size of inds positive alternative:", inds_alt['positive'].shape)
print("Number of solutions alternative:", inds_alt.shape[0])



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
# print("Num leafs: ", at_orig.num_leafs())
# print("Num leafs per tree:")
# for t in at_orig:
#     print(t.num_leaves(), end=' ')
# nb_splits_per_feature = list(map(lambda y: len(y), at_orig.get_splits().values()))
# print(at_orig.get_splits())
# print('\nNb splits per feature', nb_splits_per_feature)
# print('Bound on #solutions:', np.prod(np.array(nb_splits_per_feature) + 1))

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

# results = run_verification_tasks(at_orig, dtest.X, dtest.y, timeout=21600, n=100)
# print(results)