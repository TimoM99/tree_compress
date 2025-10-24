from functools import partial
import time
from unittest import result
from torch import neg
import veritas
from numba import njit
import numpy as np
import sys

def count_ocs(at, timeout):
    config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
    config.stop_when_optimal = False
    config.max_memory = 16*1024*1024*1024
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

def find_ocs(at, timeout):
    config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
    config.stop_when_optimal = False
    config.max_memory = 16*1024*1024*1024
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
    # import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    num_solutions = search.num_solutions()
    num_features = len(at.get_splits())

    inds = -np.ones((num_solutions, num_features), dtype=np.int32)
    doms = np.zeros((num_solutions, num_features, 2), dtype=np.float32)
    labels = np.zeros(num_solutions, dtype=np.uint8)

    s = True
    i = 0
    while s:
        try:
            sol = search.get_solution(i)
        except IndexError:
            break
            # pos_solutions.append(sol.box()) if sol.output > 0 else neg_solutions.append(sol.box())
        labels[i] = 1 if sol.output > 0 else 0

        sol = sol.box()

        k = len(sol.keys())
        inds[i, :k] = list(sol.keys())
        doms[i, :k, 0] = [dom.lo for dom in sol.values()]
        doms[i, :k, 1] = [dom.hi for dom in sol.values()]

        i += 1
    
    inds = {'positive': inds[labels == 1], 'negative': inds[labels == 0]}
    doms = {'positive': doms[labels == 1], 'negative': doms[labels == 0]}
    return inds, doms, out_of_resources, time_taken

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


def approx_emp_robustness(at, max_delta, example, target_label):
    if target_label:
        source_at, target_at = None, at
    else:
        source_at, target_at = at, None
    
    start_delta = max_delta
    rob = veritas.VeritasRobustnessSearch(
        example, start_delta, source_at, target_at, silent=True
    )
    delta, delta_lo, delta_hi = rob.search()

    return delta_lo


def exact_emp_robustness_bounded(at, max_delta, example, target_label):
    from gurobipy import GRB

    # By putting a box around the example, we make the search space smaller.
    box = [veritas.Interval(x-max_delta, x+max_delta) for x in example]
    at_pruned = at.prune(box)
    kan = veritas.KantchelianAttack(at_pruned, target_label, example)
    kan.model.setParam(GRB.Param.TimeLimit, 10*60.0)
    kan.model.setParam(GRB.Param.Threads, 1)
    kan.optimize()
    try:
        return min(max_delta, kan.bounds[-1][0])
    except IndexError:
        return max_delta

    #def linf(x, y):
    #    return np.max(np.abs(x-y))

    #return linf(example, kan.solution()[0])

def exact_emp_robustness(at, example, target_label):
    from gurobipy import GRB

    kan = veritas.KantchelianAttack(at, target_label, example)
    kan.model.setParam(GRB.Param.TimeLimit, 10*60.0)
    kan.model.setParam(GRB.Param.Threads, 1)
    kan.optimize()
    # print(kan.bounds)
    try:
        return kan.bounds[-1][0]
    except IndexError:
        return 1e18

def emp_robustness_linear_scan(inds, doms, example, target_label):
    if target_label:
        inds = inds['positive']
        doms = doms['positive']
    else:
        inds = inds['negative']
        doms = doms['negative']

    min_dist = min_dist_to_solutions(example, inds, doms, len(inds))

    return min_dist

# def get_inds_doms(pos_solutions, neg_solutions):
#     pos_solutions = [(list(sol.keys()), [(dom.lo, dom.hi) for dom in sol.values()]) for sol in pos_solutions]
#     max_k = max(len(sol[0]) for sol in pos_solutions)
#     n_pos = len(pos_solutions)

#     pos_inds = -np.ones((n_pos, max_k), dtype=np.int32)
#     pos_doms = np.zeros((n_pos, max_k, 2), dtype=np.float32)

#     for i, sol in enumerate(pos_solutions):
#         k = len(sol[0])
#         pos_inds[i, :k] = sol[0]
#         pos_doms[i, :k] = sol[1]

#     neg_solutions = [(list(sol.keys()), [(dom.lo, dom.hi) for dom in sol.values()]) for sol in neg_solutions]
#     max_k = max(len(sol[0]) for sol in neg_solutions)
#     n_neg = len(neg_solutions)

#     neg_inds = -np.ones((n_neg, max_k), dtype=np.int32)
#     neg_doms = np.zeros((n_neg, max_k, 2), dtype=np.float32)

#     for i, sol in enumerate(neg_solutions):
#         k = len(sol[0])
#         neg_inds[i, :k] = sol[0]
#         neg_doms[i, :k] = sol[1]

#     inds = {'positive': pos_inds, 'negative': neg_inds}
#     doms = {'positive': pos_doms, 'negative': neg_doms}

#     return inds, doms

def emp_robustness(at, x, y, n, method, timeout):
    delta_lo = 0.0
    count = 0
    result = {}
    if method == 'exact':
        f = partial(exact_emp_robustness, at)
    elif method == 'approx':
        f = partial(approx_emp_robustness, at, 1.0)
    elif method == 'linear_scan':
        inds, doms, out_of_resources, time_taken = find_ocs(at, timeout)
        # print(sys.getsizeof(inds), sys.getsizeof(doms))
        # print(inds['positive'].shape)
        # print(inds['positive'].size*inds['positive'].itemsize)
        # print(doms['positive'].size*doms['positive'].itemsize)
        # print(sys.getsizeof(pos_solutions) + sys.getsizeof(neg_solutions))
        # import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        # inds, doms = get_inds_doms(pos_solutions, neg_solutions)
        # print(sys.getsizeof(inds) + sys.getsizeof(doms))
        # import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        f = partial(emp_robustness_linear_scan, inds, doms)
        result['oc_space'] = len(inds['positive']) + len(inds['negative'])
        result['failed_oc'] = out_of_resources
        result['time_taken_oc'] = time_taken
    t = time.time()
    for i in x.index:
        target_label = not (y.loc[i] > 0.0)
        example = x.loc[i, :].to_numpy()
        pred_label = at.eval(example)[0, 0] > 0.0

        if pred_label != target_label:
            res = f(example, target_label)
            delta_lo += res

        count += 1
        if count >= n:
            break
        if time.time() - t > timeout:
            break
    t = time.time() - t
    result['emp_rob'] = delta_lo / count
    result['emp_rob_n'] = count
    result['emp_rob_time'] = t
    return result


def constrast_two_examples(at):
    splits = at.get_splits()
    n_features = max(splits.keys()) + 1
    columns = [f"F{i}" for i in range(n_features)]
    nonfixed = sorted(splits.items(), key=lambda p: len(p[1]))[-1][0]

    feat_map = veritas.FeatMap(columns)
    for k, column in enumerate(columns):
        if k == nonfixed:
            index_for_instance0 = feat_map.get_index(column, 0)
            index_for_instance1 = feat_map.get_index(column, 1)
            feat_map.use_same_id_for(index_for_instance0, index_for_instance1)

    at_for_instance1 = feat_map.transform(at, 1)
    at_contrast = at.concat_negated(at_for_instance1)

    return at_contrast, feat_map

def fairness_task(at, timeout):
    t = time.time()
    at_contrast, feat_map = constrast_two_examples(at)
    config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)

    config.ignore_state_when_worse_than = 0.0
    config.focal_eps = 0.95
    config.max_focal_size = 100
    config.max_memory = 16*1024*1024*1024

    search = config.get_search(at_contrast)

    num_search_steps_per_iteration = 1000
    has_timed_out = False
    oom = False

    while True:
        stop_reason = search.steps(num_search_steps_per_iteration)
        if stop_reason == veritas.StopReason.NO_MORE_OPEN:
            break
        if stop_reason == veritas.StopReason.OUT_OF_MEMORY:
            oom = True
            break
        if search.num_solutions() > 0:
            break
        if search.time_since_start() > timeout:
            has_timed_out = True
            break

    #bound_lh = search.current_bounds()
    #isfair = bound_lh <= 0.0

    t = time.time() - t
    isfair = search.num_solutions() > 0

    return isfair, has_timed_out or oom, t

def run_verification_tasks(at, x, y, timeout, n):

    ## VERIFICATION: (1) HOW MANY OCs?
    # nocs, nocs_time, nocs_timeout = count_ocs(at, timeout)

    # VERIFICATION: (2) Empricial robustness (exact + approx)
    result_exact_emp_rob = emp_robustness(
        at, x, y, n, method='exact', timeout=timeout
    )
    result_approx_emp_rob = emp_robustness(
        at, x, y, n, method='approx', timeout=timeout
    )
    result_exact_emp_rob_linear_scan = emp_robustness(
        at, x, y, n, method='linear_scan', timeout=timeout
    )




    # isfair, fair_timeout, fair_time = fairness_task(at, timeout)

    return {
        # "nocs": nocs,
        # "nocs_time": nocs_time,
        # "nocs_timeout": nocs_timeout,
        "exact_emp_rob": result_exact_emp_rob['emp_rob'],
        "exact_emp_rob_n": result_exact_emp_rob['emp_rob_n'],
        "exact_emp_rob_timeout": result_exact_emp_rob['emp_rob_time'] >= timeout,
        "exact_emp_rob_time": result_exact_emp_rob['emp_rob_time'],
        "approx_emp_rob": result_approx_emp_rob['emp_rob'],
        "approx_emp_rob_n": result_approx_emp_rob['emp_rob_n'],
        "approx_emp_rob_timeout": result_approx_emp_rob['emp_rob_time'] >= timeout,
        "approx_emp_rob_time": result_approx_emp_rob['emp_rob_time'],
        "exact_emp_rob_linear_scan": result_exact_emp_rob_linear_scan['emp_rob'],
        "exact_emp_rob_linear_scan_n": result_exact_emp_rob_linear_scan['emp_rob_n'],
        "exact_emp_rob_linear_scan_timeout": result_exact_emp_rob_linear_scan['emp_rob_time'] >= timeout,
        "exact_emp_rob_linear_scan_time": result_exact_emp_rob_linear_scan['emp_rob_time'],
        "oc_space": result_exact_emp_rob_linear_scan['oc_space'],
        "failed_oc": result_exact_emp_rob_linear_scan['failed_oc'],
        # "isfair": isfair,
        # "fair_timeout": fair_timeout,
        # "fair_time": fair_time,
    }
