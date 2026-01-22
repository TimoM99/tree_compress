from functools import partial
import time
from unittest import result
from sklearn.preprocessing import TargetEncoder
from torch import ne, neg
import veritas
from numba import njit
import numpy as np
import sys
import h5py
import index as index

import resource
from multiprocessing import Process, Pipe
import traceback


# ------------------------------
# Worker-side memory limit
# ------------------------------
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

# def find_ocs(at, timeout, memory_limit):
#     config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
#     config.stop_when_optimal = False
#     config.max_memory = memory_limit
#     search = config.get_search(at)

#     has_timed_out = False
#     oom = False

#     while True:
#         stop_reason = search.steps(1000)
#         # print(search.get_used_memory())
#         if stop_reason == veritas.StopReason.NO_MORE_OPEN:
#             break

#         has_timed_out = search.time_since_start() >= timeout
#         oom = stop_reason == veritas.StopReason.OUT_OF_MEMORY
#         if has_timed_out or oom:
#             break

#     out_of_resources = has_timed_out or oom
#     time_taken = search.time_since_start()
#     memory_used = search.get_used_memory()

#     s = True
#     i = 0
#     pos_count = 0
#     max_box_size = 0
#     while s:
#         try:
#             sol = search.get_solution(i)
#         except IndexError:
#             break
        
#         pos_count += 1 if sol.output > 0 else 0
#         max_box_size = max(max_box_size, len(sol.box().keys()))
#         i += 1

#     num_solutions = search.num_solutions()
#     # num_features = len(at.get_splits())

#     pos_inds = -np.ones((pos_count, max_box_size), dtype=np.int32)
#     neg_inds = -np.ones((num_solutions - pos_count, max_box_size), dtype=np.int32)
#     pos_doms = np.zeros((pos_count, max_box_size, 2), dtype=np.float32)
#     neg_doms = np.zeros((num_solutions - pos_count, max_box_size, 2), dtype=np.float32)

#     p = 0
#     n = 0
#     while s:
#         try:
#             sol = search.get_solution(p+n)
#         except IndexError:
#             break
#             # pos_solutions.append(sol.box()) if sol.output > 0 else neg_solutions.append(sol.box())
#         label = 1 if sol.output > 0 else 0

#         sol = sol.box()
#         k = len(sol.keys())

#         if label == 1:
#             pos_inds[p, :k] = list(sol.keys())
#             pos_doms[p, :k, 0] = [dom.lo for dom in sol.values()]
#             pos_doms[p, :k, 1] = [dom.hi for dom in sol.values()]
#             p += 1

#         else:
#             neg_inds[n, :k] = list(sol.keys())
#             neg_doms[n, :k, 0] = [dom.lo for dom in sol.values()]
#             neg_doms[n, :k, 1] = [dom.hi for dom in sol.values()]
#             n += 1
        

#     inds = {'positive': pos_inds, 'negative': neg_inds}
#     doms = {'positive': pos_doms, 'negative': neg_doms}
#     return inds, doms, out_of_resources, time_taken, memory_used

def find_ocs(at, timeout, memory_limit):
    start_time = time.time()
    out_of_resources = False
    memory_limit_reached = False

    doms = np.full((0,0,2), 0)
    inds = np.full((0,0), 0)
    preds = np.full((0,), 0)
    try:
        inds, doms, preds = run_with_timeout_and_memory(
            enumerate_ocs, at, timeout=timeout, mem_limit=memory_limit
        )
    except (TimeoutError, MemoryError) as e:
        out_of_resources = True
        memory_limit_reached = isinstance(e, MemoryError)

    
    time_taken = time.time() - start_time

    return inds, doms, preds, out_of_resources, time_taken, memory_limit_reached

def enumerate_ocs(at):
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
            doms, inds, preds = cross_product(doms, inds, preds, leaf_doms, leaf_inds, leaf_preds)

    return inds, doms, preds


@njit
def cross_product(doms, inds, preds, leaf_doms, leaf_inds, leaf_preds):
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
                max_size = max(max_size, size) #Keep track of the maximum number of OC features used.
    doms_result = np.zeros((count, max_size, 2), dtype=np.float32)
    inds_result = -np.ones((count, max_size), dtype=np.int32)
    preds_result = np.zeros((count,), dtype=np.float32)
    
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
                # Merge dom and leaf_dom
                merge_doms(dom, ind, leaf_dom, leaf_ind, doms_result[k], inds_result[k])
                preds_result[k] = pred + leaf_pred
                k += 1
    
    return doms_result, inds_result, preds_result

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
    i = 0
    j = 0
    l = 0
    while i < len(leaf_ind) and j < len(ind):
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



@njit
def dist_to_boxes(example, boxes, feat_map):
    # _boxes = boxes[(preds > 0.0) == target_label]
    dist = np.zeros((boxes.shape[0],), dtype=np.float64)
    for s in range(boxes.shape[0]):
        max_dist = 0.0
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
            # idx = feat_map[i]
            # print(idx)
            # if idx == -1:
            #     continue
            x = example[i]
            lo, hi = los[idx], his[idx]

            d = 0
            t = lo - x
            if t > 0.0:
                d = t
            t = x - hi
            if t > d:
                d = t

            if d > max_dist:
                max_dist = d
            # if x < lo:
            #     d = lo - x
            # elif x > hi:
            #     d = x - hi
            # else:
            #     d = 0.0
            # if d > max_dist:
            #     max_dist = d
        dist[s] = max_dist
    return dist

@njit
def min_dist_to_solutions(example, boxes, feat_map):
    # _boxes = boxes[(preds > 0.0) == target_label]
    min_dist = 1e18
    for s in range(boxes.shape[0]):
        max_dist = 0.0
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
            # idx = feat_map[i]
            # print(idx)
            # if idx == -1:
            #     continue
            x = example[i]
            lo, hi = los[idx], his[idx]

            d = 0
            t = lo - x
            if t > 0.0:
                d = t
            t = x - hi
            if t > d:
                d = t

            if d > max_dist:
                max_dist = d
                if max_dist >= min_dist:
                    break
            # if x < lo:
            #     d = lo - x
            # elif x > hi:
            #     d = x - hi
            # else:
            #     d = 0.0
            # if d > max_dist:
            #     max_dist = d
        if max_dist < min_dist:
            min_dist = max_dist
    return min_dist


def approx_emp_robustness(at, max_delta, x, y, time_limit):
    t0 = time.time()
    
    delta_lo = np.zeros(x.shape[0])
    verification_times = np.zeros(x.shape[0])
    count = 0
    
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()
        if target_label:
            source_at, target_at = None, at
        else:
            source_at, target_at = at, None
        
        start_delta = max_delta
        rob = veritas.VeritasRobustnessSearch(
            example, start_delta, source_at, target_at, silent=True
        )
        _, _delta_lo, _ = rob.search()
        
        delta_lo[i] = _delta_lo
        verification_times[i] = time.time() - t
        count += 1

        if time.time() - t0 >= time_limit:
            break

    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'verification_times': verification_times.tolist(), 'robustness_values': delta_lo.tolist()}


# def exact_emp_robustness_bounded(at, max_delta, example, target_label):
#     from gurobipy import GRB

#     # By putting a box around the example, we make the search space smaller.
#     box = [veritas.Interval(x-max_delta, x+max_delta) for x in example]
#     at_pruned = at.prune(box)
#     kan = veritas.KantchelianAttack(at_pruned, target_label, example)
#     kan.model.setParam(GRB.Param.TimeLimit, 10*60.0)
#     kan.model.setParam(GRB.Param.Threads, 1)
#     kan.optimize()
#     try:
#         return min(max_delta, kan.bounds[-1][0])
#     except IndexError:
#         return max_delta

#     #def linf(x, y):
#     #    return np.max(np.abs(x-y))

#     #return linf(example, kan.solution()[0])

def exact_emp_robustness(at, x, y, time_limit):
    from gurobipy import GRB

    t0 = time.time()
    delta_lo = np.zeros(x.shape[0], np.float64)
    verification_times = np.zeros(x.shape[0])
    count = 0

    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()
        kan = veritas.KantchelianAttack(at, target_label, example)
        kan.model.setParam(GRB.Param.TimeLimit, time_limit - (time.time() - t))
        kan.model.setParam(GRB.Param.Threads, 1)
        kan.optimize()
        
        try:
            delta_lo[i] = kan.bounds[-1][0]
        except IndexError:
            delta_lo[i] = 1e18
        count += 1
        verification_times[i] = time.time() - t
        if np.sum(verification_times) >= time_limit:
            break

    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'verification_times': verification_times.tolist(), 'robustness_values': delta_lo.tolist()}

def emp_robustness_linear_scan(oc_file, x, y, time_limit):
    t0 = time.time()
    f = h5py.File(oc_file, "r") #cache size doesn't matter because we are reading sequentially

    boxes = f['boxes']
    preds = f['outvalues']


    num_boxes = f.attrs['num_solutions']
    feat_map = f.attrs['feat_map']

    active_map = np.where(feat_map != -1)[0]
    feat_map = feat_map[active_map]

    buffer_size = boxes.chunks[0]

    # print(len(feat_map))

    total_bytes = 0 
    time_reading = 0.0
    verification_times = np.zeros(x.shape[0])
    count = 500

    _boxes = np.zeros((buffer_size, boxes.shape[1], boxes.shape[2]), dtype=boxes.dtype)
    _preds = np.zeros((buffer_size,), dtype=preds.dtype)

    min_dist_arr = np.full((x.shape[0],), 1e18)
    for i in range(0, num_boxes, buffer_size):
        t = time.perf_counter()
        # print('reading')
        n = min(buffer_size, num_boxes - i)
        boxes.read_direct(_boxes[:n], np.s_[i:i+n])
        preds.read_direct(_preds[:n], np.s_[i:i+n])

        time_reading += time.perf_counter() - t
        total_bytes += _boxes[:n].nbytes + _preds[:n].nbytes
        # print(_preds)

        pos_boxes = _boxes[:n][(_preds[:n] > 0.0)]

        neg_boxes = _boxes[:n][(_preds[:n] <= 0.0)]
        # print(neg_boxes)
        # print('calculating')
        for i, (example, target_label) in enumerate(zip(x, y)):
            # print(example, target_label)
            t = time.perf_counter()
            if target_label:
                boxes_to_consider = pos_boxes
            else:
                boxes_to_consider = neg_boxes
            min_dist_arr[i] = min(min_dist_arr[i], min_dist_to_solutions(example, boxes_to_consider, active_map))
            verification_times[i] += time.perf_counter() - t
        # dist = min_dist_to_solutions(x, _boxes[:n], _preds[:n], y, feat_map)
        # print(dist)
    
        if time_reading + np.sum(verification_times) >= time_limit:
            count = None
            break

    # print(f"Scanned {num_boxes} boxes in {time_reading:.2f} seconds (reading), {np.sum(verification_times):.2f} seconds (calculating), {total_bytes / (1024**2):.2f} MB read")


    return {'emp_rob': np.mean(min_dist_arr), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'time_reading': time_reading, 'verification_times': verification_times.tolist(), 'robustness_values': min_dist_arr.tolist()}

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

def emp_robustness_rootbox_index(oc_file, x, y, time_limit):
    t0 = time.time()
    rbi = index.RootboxIndex.load(oc_file)
    f = h5py.File(oc_file, "r") #cache size doesn't matter because we are reading sequentially

    boxes = f['boxes']
    preds = f['outvalues']
    feat_map = rbi.featmap
    active_map = np.where(feat_map != -1)[0]

    buffer_size = boxes.chunks[0]
    reading_time = 0.0
    
    delta_lo = np.full(x.shape[0], 1e18)
    verification_times = np.zeros(x.shape[0])

    count = 0
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()
        idx = rbi.find_index(example)
        boxes_ids = rbi.get_boxes_at_index(idx)
        verification_times[i] += time.time() - t
        num_solutions = boxes_ids.shape[0]

        # print(example)
        # print(target_label)
        _boxes = np.zeros((buffer_size, boxes.shape[1], boxes.shape[2]), dtype=boxes.dtype)
        _preds = np.zeros((buffer_size,), dtype=preds.dtype)
        for j in range(0, num_solutions, buffer_size):
            n = min(buffer_size, num_solutions - j)

            t = time.time()
            boxes.read_direct(_boxes[:n], np.s_[boxes_ids[j:j+n]])
            preds.read_direct(_preds[:n], np.s_[boxes_ids[j:j+n]])
            reading_time += time.time() - t

            pos_boxes = _boxes[:n][(_preds[:n] > 0.0)]
            neg_boxes = _boxes[:n][(_preds[:n] <= 0.0)]

            t = time.time()
            delta_lo[i] = min(delta_lo[i], min_dist_to_solutions(example, pos_boxes if target_label else neg_boxes, active_map))
            verification_times[i] += time.time() - t

        t = time.time()
        deltas = dist_to_boxes(example, rbi.get_rootboxes(), active_map)
        sorted_rootboxes = np.argsort(deltas)
        verification_times[i] += time.time() - t
        # print(example.dtype)
        # print(delta_lo[i])
        # print(delta_lo[i].dtype)
        # print(deltas)
        # print(deltas[3])
        # print(deltas)
        for ibox in sorted_rootboxes[1:]: #skip the first one, already considered
            # print(deltas[ibox], delta_lo[i])
            if deltas[ibox] > delta_lo[i]:
                continue
            # print('we get here')
            t = time.time()
            boxes_ids = rbi.get_boxes_at_index(ibox)
            verification_times[i] += time.time() - t
            # print(boxes_ids)
            num_solutions = boxes_ids.shape[0]

            for j in range(0, num_solutions, buffer_size):
                n = min(buffer_size, num_solutions - j)

                t = time.time()
                boxes.read_direct(_boxes[:n], np.s_[boxes_ids[j:j+n]])
                preds.read_direct(_preds[:n], np.s_[boxes_ids[j:j+n]])
                reading_time += time.time() - t

                # print(_boxes)

                # print('checking boxes in rootbox', ibox)

                pos_boxes = _boxes[:n][(_preds[:n] > 0.0)]
                # print(neg_boxes)
                neg_boxes = _boxes[:n][(_preds[:n] <= 0.0)]
                # print(neg_boxes)

                t = time.time()
                # print(delta_lo[i], min_dist_to_solutions(example, pos_boxes if target_label else neg_boxes, active_map))
                delta_lo[i] = min(delta_lo[i], min_dist_to_solutions(example, pos_boxes if target_label else neg_boxes, active_map))
                verification_times[i] += time.time() - t
            # print(delta_lo[i])

        if time.time() - t0 >= time_limit:
            break
        count += 1
    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'verification_times': verification_times.tolist(), 'robustness_values': delta_lo.tolist(), 'time_reading': reading_time}



def emp_robustness(at, x, y, n, method, time_limit, oc_file=None):
    # delta_lo = 0.0
    # count = 0
    # result = {}
    
        # inds, doms, preds, out_of_resources, time_taken, memory_limit_reached = find_ocs(at, timeout, memory_limit)
        # print(sys.getsizeof(inds), sys.getsizeof(doms))
        # print(inds['positive'].shape)
        # print(inds['positive'].size*inds['positive'].itemsize)
        # print(doms['positive'].size*doms['positive'].itemsize)
        # print(sys.getsizeof(pos_solutions) + sys.getsizeof(neg_solutions))
        # import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        # inds, doms = get_inds_doms(pos_solutions, neg_solutions)
        # print(sys.getsizeof(inds) + sys.getsizeof(doms))
        # import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        # f = partial(emp_robustness_linear_scan, oc_file)
    # t = time.time()
    x_correct = []
    y_correct = []
    count = 0
    for i in x.index:
        target_label = not (y.loc[i] > 0.0)
        example = x.loc[i, :].to_numpy()
        pred_label = at.eval(example)[0, 0] > 0.0

        if pred_label != target_label:
            x_correct.append(example)
            y_correct.append(target_label)

            count += 1
        if count >= n:
            break
    x_correct = np.array(x_correct)
    y_correct = np.array(y_correct)
    if method == 'exact':
        result = exact_emp_robustness(at, x_correct, y_correct, time_limit)
    elif method == 'approx':
        result = approx_emp_robustness(at, 1.0, x_correct, y_correct, time_limit)
    elif method == 'linear_scan':
        assert oc_file is not None
        result = emp_robustness_linear_scan(oc_file, x_correct, y_correct, time_limit)
    elif method == 'rootbox_index':
        result = emp_robustness_rootbox_index(oc_file, x_correct, y_correct, time_limit)
    # t = time.time() - t
    # result['emp_rob'] = delta_lo / count
    # result['emp_rob_n'] = count
    # result['emp_rob_time'] = t
    result['count'] = count
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

def run_verification_tasks(at, x, y, oc_file, timeout, n):

    ## VERIFICATION: (1) HOW MANY OCs?
    # nocs, nocs_time, nocs_timeout = count_ocs(at, timeout)

    # VERIFICATION: (2) Empricial robustness (exact + approx)
    result_exact_emp_rob_linear_scan = emp_robustness(
        at, x, y, n, method='linear_scan', time_limit=timeout, oc_file=oc_file
    )
    result_exact_emp_rob_rootbox_index = emp_robustness(
        at, x, y, n, method='rootbox_index', time_limit=timeout, oc_file=oc_file
    )
    result_exact_emp_rob = emp_robustness(
        at, x, y, n, method='exact', time_limit=timeout
    )
    result_approx_emp_rob = emp_robustness(
        at, x, y, n, method='approx', time_limit=timeout
    )




    # isfair, fair_timeout, fair_time = fairness_task(at, timeout)

    return {
        "exact_emp_rob": result_exact_emp_rob['emp_rob'],
        "exact_emp_rob_n": result_exact_emp_rob['emp_rob_n'],
        "exact_emp_rob_timeout": result_exact_emp_rob['emp_rob_time'] >= timeout,
        "exact_emp_rob_time": result_exact_emp_rob['emp_rob_time'],
        "exact_emp_rob_verification_times": result_exact_emp_rob['verification_times'],
        "exact_emp_rob_robustness_values": result_exact_emp_rob['robustness_values'],
        "approx_emp_rob": result_approx_emp_rob['emp_rob'],
        "approx_emp_rob_n": result_approx_emp_rob['emp_rob_n'],
        "approx_emp_rob_timeout": result_approx_emp_rob['emp_rob_time'] >= timeout,
        "approx_emp_rob_time": result_approx_emp_rob['emp_rob_time'],
        "approx_emp_rob_verification_times": result_approx_emp_rob['verification_times'],
        "approx_emp_rob_robustness_values": result_approx_emp_rob['robustness_values'],
        "exact_emp_rob_linear_scan": result_exact_emp_rob_linear_scan['emp_rob'],
        "exact_emp_rob_linear_scan_n": result_exact_emp_rob_linear_scan['emp_rob_n'],
        "exact_emp_rob_linear_scan_timeout": result_exact_emp_rob_linear_scan['emp_rob_time'] >= timeout,
        "exact_emp_rob_linear_scan_time": result_exact_emp_rob_linear_scan['emp_rob_time'],
        "exact_emp_rob_linear_scan_reading_time": result_exact_emp_rob_linear_scan['time_reading'],
        "exact_emp_rob_linear_scan_verification_times": result_exact_emp_rob_linear_scan['verification_times'],
        "exact_emp_rob_linear_scan_robustness_values": result_exact_emp_rob_linear_scan['robustness_values'],
        "exact_emp_rob_rootbox_index": result_exact_emp_rob_rootbox_index['emp_rob'],
        "exact_emp_rob_rootbox_index_n": result_exact_emp_rob_rootbox_index['emp_rob_n'],
        "exact_emp_rob_rootbox_index_timeout": result_exact_emp_rob_rootbox_index['emp_rob_time'] >= timeout,
        "exact_emp_rob_rootbox_index_time": result_exact_emp_rob_rootbox_index['emp_rob_time'],
        "exact_emp_rob_rootbox_index_reading_time": result_exact_emp_rob_rootbox_index['time_reading'],
        "exact_emp_rob_rootbox_index_verification_times": result_exact_emp_rob_rootbox_index['verification_times'],
        "exact_emp_rob_rootbox_index_robustness_values": result_exact_emp_rob_rootbox_index['robustness_values'],
        # "nocs
        # "oc_space": result_exact_emp_rob_linear_scan['oc_space'],
        # "failed_oc": result_exact_emp_rob_linear_scan['failed_oc'],
        # "time_taken_oc": result_exact_emp_rob_linear_scan['time_taken_oc'],
        # "memory_limit_reached": result_exact_emp_rob_linear_scan['memory_limit_reached'],
    }
