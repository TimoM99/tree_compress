from functools import partial
from math import dist, e
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

def hdf5_generator(file_path, dataset_name, batch_size):
    """Yields batches of data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name]
        num_samples = data.shape[0]
        for i in range(0, num_samples, batch_size):
            # Read only the necessary chunk into memory
            yield data[i:i + batch_size]

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

@njit
def dist_to_boxes(example, boxes, feat_map):
    dist = np.zeros((boxes.shape[0],), dtype=np.float64)
    for s in range(boxes.shape[0]):
        max_dist = 0.0
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
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
        dist[s] = max_dist
    return dist

@njit
def min_dist_to_solutions(example, boxes, feat_map):
    min_dist = 1e18
    for s in range(boxes.shape[0]):
        max_dist = 0.0
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
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

def emp_robustness_linear_scan_disk(oc_file, x, y, time_limit):
    t0 = time.time()
    f = h5py.File(oc_file, "r") #cache size doesn't matter because we are reading sequentially

    boxes = f['boxes']
    preds = f['outvalues']


    num_boxes = f.attrs['num_solutions']
    feat_map = f.attrs['feat_map']

    active_map = np.where(feat_map != -1)[0]

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

def emp_robustness_linear_scan(at, oc_file, x, y, time_limit):
    t0 = time.time()
    boxes = hdf5_generator(oc_file, 'boxes', 8194)
    preds = hdf5_generator(oc_file, 'outvalues', 8194)
    with h5py.File(oc_file, "r") as f:
        feat_map = f.attrs['feat_map']
    active_map = np.where(feat_map != -1)[0]
    pni = index.PosNegIndex(at, feat_map)
    for boxes_batch, preds_batch in zip(boxes, preds):
        pni.store(boxes_batch, preds_batch)
    pni.to_arrays()



    index_time = time.time() - t0
    t0 = time.time()

    
    delta_lo = np.full(x.shape[0], 1e18)
    verification_times = np.zeros(x.shape[0])
    count = 0

    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()

        boxes = pni.get_boxes_at_index(target_label)
        delta_lo[i] = min_dist_to_solutions(example, boxes, active_map)
        verification_times[i] = time.time() - t
        count += 1

        if time.time() - t0 >= time_limit:
            break
    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'verification_times': verification_times.tolist(), 'robustness_values': delta_lo.tolist(), 'index_time': index_time}


def emp_robustness_rootbox_index_disk(oc_file, x, y, time_limit):
    t0 = time.time()
    rbi = index.RootboxIndex.load(oc_file)
    f = h5py.File(oc_file, "r", rdcc_nbytes=1024**3, rdcc_nslots=10_000_000) #cache size doesn't matter because we are reading sequentially

    boxes = f['boxes']
    preds = f['outvalues']
    feat_map = rbi.featmap
    active_map = np.where(feat_map != -1)[0]

    buffer_size = boxes.chunks[0]
    reading_time = 0.0
    
    delta_lo = np.full(x.shape[0], 1e18)
    verification_times = np.zeros(x.shape[0])

    #TODO Extract reading process
    count = 0
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()
        idx = rbi.find_index(example)
        boxes_ids = rbi.get_boxes_at_index(idx)
        # boxes_ids = np.sort(boxes_ids)
        verification_times[i] += time.time() - t
        num_solutions = boxes_ids.shape[0]

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

        for ibox in sorted_rootboxes[1:]: #skip the first one, already considered
            if deltas[ibox] > delta_lo[i]:
                break

            t = time.time()
            boxes_ids = rbi.get_boxes_at_index(ibox)
            # boxes_ids = np.sort(boxes_ids)
            verification_times[i] += time.time() - t

            num_solutions = boxes_ids.shape[0]

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

        if time.time() - t0 >= time_limit:
            break
        count += 1
    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'verification_times': verification_times.tolist(), 'robustness_values': delta_lo.tolist(), 'time_reading': reading_time}

def emp_robustness_rootbox_index(at, oc_file, x, y, time_limit):
    t0 = time.time()
    boxes = hdf5_generator(oc_file, 'boxes', 100*8194)
    preds = hdf5_generator(oc_file, 'outvalues', 100*8194)
    with h5py.File(oc_file, "r") as f:
        feat_map = f.attrs['feat_map']
    active_map = np.where(feat_map != -1)[0]

    rbi = index.RootboxIndex(at, feat_map)
    for boxes_batch, preds_batch in zip(boxes, preds):
        rbi.store(boxes_batch, preds_batch)
    rbi.to_arrays()
    
    index_time = time.time() - t0
    t0 = time.time()

    delta_lo = np.full(x.shape[0], 1e18)
    verification_times = np.zeros(x.shape[0])
    count = 0

    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()

        idx = rbi.find_index(example)
        boxes = rbi.get_boxes_at_index(idx, target_label)

        # TODO use generator
        delta_lo[i] = min_dist_to_solutions(example, boxes, active_map)

        deltas = dist_to_boxes(example, rbi.get_rootboxes(), active_map)
        sorted_rootboxes = np.argsort(deltas)

        for ibox in sorted_rootboxes: #skip the first one, already considered
            if deltas[ibox] > delta_lo[i]:
                break

            boxes = rbi.get_boxes_at_index(ibox, target_label)
            delta_lo[i] = min(delta_lo[i], min_dist_to_solutions(example, boxes, active_map))

        verification_times[i] = time.time() - t
        count += 1

        if time.time() - t0 >= time_limit:
            break

    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'verification_times': verification_times.tolist(), 'robustness_values': delta_lo.tolist(), 'index_time': index_time}

def emp_robustness_oc_index(at, oc_file, x, y, time_limit):
    t0 = time.time()
    boxes = hdf5_generator(oc_file, 'boxes', 100*8194)
    preds = hdf5_generator(oc_file, 'outvalues', 100*8194)
    with h5py.File(oc_file, "r") as f:
        feat_map = f.attrs['feat_map']
    active_map = np.where(feat_map != -1)[0]
    # print('init')
    oci = index.OCIndex(at, feat_map)
    # print('storing')
    # r = 0
    for boxes_batch, preds_batch in zip(boxes, preds):
        oci.store(boxes_batch, preds_batch)
        # r += boxes_batch.shape[0]
        # print(r)
    # print('to arrays')
    oci.to_arrays()

    index_time = time.time() - t0
    t0 = time.time()

    delta_lo = np.full(x.shape[0], 1e18)
    verification_times = np.zeros(x.shape[0])
    count = 0

    
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()

        boxes = oci.get_boxes(example, target_label)

        delta_lo[i] = min_dist_to_solutions(example, boxes, active_map)

        q = [idx for idx in oci.index]
        while q != []:
            index_entry = q.pop()
            if delta_lo[i] < dist_to_boxes(example, np.array([index_entry.box]), active_map)[0]:
                continue
            
            if index_entry.pos is not None and index_entry.neg is not None:
                boxes_to_consider = index_entry.pos if target_label else index_entry.neg
                delta_lo[i] = min(delta_lo[i], min_dist_to_solutions(example, boxes_to_consider, active_map))

            else:
                for child in index_entry.children:
                    q.append(child)

        verification_times[i] = time.time() - t
        count += 1

        if time.time() - t0 >= time_limit:
            break

    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'verification_times': verification_times.tolist(), 'robustness_values': delta_lo.tolist(), 'index_time': index_time}

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
        result = emp_robustness_linear_scan(at, oc_file, x_correct, y_correct, time_limit)
    elif method == 'rootbox_index':
        result = emp_robustness_rootbox_index(at, oc_file, x_correct, y_correct, time_limit)
    elif method == 'oc_index':
        result = emp_robustness_oc_index(at, oc_file, x_correct, y_correct, time_limit)


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
    # print('starting rootbox index')
    result_exact_emp_rob_rootbox_index = emp_robustness(
        at, x, y, n, method='rootbox_index', time_limit=timeout, oc_file=oc_file
    )
    result_exact_emp_rob_oc_index = emp_robustness(
        at, x, y, n, method='oc_index', time_limit=timeout, oc_file=oc_file
    ) if len(at) > 2 else emp_robustness(at, x, y, n, method='linear_scan', time_limit=timeout, oc_file=oc_file)

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
        "exact_emp_rob_linear_scan_verification_times": result_exact_emp_rob_linear_scan['verification_times'],
        "exact_emp_rob_linear_scan_robustness_values": result_exact_emp_rob_linear_scan['robustness_values'],
        "exact_emp_rob_rootbox_index": result_exact_emp_rob_rootbox_index['emp_rob'],
        "exact_emp_rob_rootbox_index_n": result_exact_emp_rob_rootbox_index['emp_rob_n'],
        "exact_emp_rob_rootbox_index_timeout": result_exact_emp_rob_rootbox_index['emp_rob_time'] >= timeout,
        "exact_emp_rob_rootbox_index_time": result_exact_emp_rob_rootbox_index['emp_rob_time'],
        "exact_emp_rob_rootbox_index_verification_times": result_exact_emp_rob_rootbox_index['verification_times'],
        "exact_emp_rob_rootbox_index_robustness_values": result_exact_emp_rob_rootbox_index['robustness_values'],
        "exact_emp_rob_rootbox_index_index_time": result_exact_emp_rob_rootbox_index['index_time'],
        "exact_emp_rob_oc_index": result_exact_emp_rob_oc_index['emp_rob'],
        "exact_emp_rob_oc_index_n": result_exact_emp_rob_oc_index['emp_rob_n'],
        "exact_emp_rob_oc_index_timeout": result_exact_emp_rob_oc_index['emp_rob_time'] >= timeout,
        "exact_emp_rob_oc_index_time": result_exact_emp_rob_oc_index['emp_rob_time'],
        "exact_emp_rob_oc_index_verification_times": result_exact_emp_rob_oc_index['verification_times'],
        "exact_emp_rob_oc_index_robustness_values": result_exact_emp_rob_oc_index['robustness_values'],
        "exact_emp_rob_oc_index_index_time": result_exact_emp_rob_oc_index['index_time'],

        # "nocs
        # "oc_space": result_exact_emp_rob_linear_scan['oc_space'],
        # "failed_oc": result_exact_emp_rob_linear_scan['failed_oc'],
        # "time_taken_oc": result_exact_emp_rob_linear_scan['time_taken_oc'],
        # "memory_limit_reached": result_exact_emp_rob_linear_scan['memory_limit_reached'],
    }

def adv_robustness(at, oc_file, x, y, n, l_inf, l_1, time_limit):
    t0 = time.time()
    boxes = hdf5_generator(oc_file, 'boxes', 100*8194)
    preds = hdf5_generator(oc_file, 'outvalues', 100*8194)
    with h5py.File(oc_file, "r") as f:
        feat_map = f.attrs['feat_map']
    active_map = np.where(feat_map != -1)[0]

    rbi = index.RootboxIndex(at, feat_map)
    for boxes_batch, preds_batch in zip(boxes, preds):
        rbi.store(boxes_batch, preds_batch)
    rbi.to_arrays()
    
    index_time = time.time() - t0
    t0 = time.time()

    sat = 0
    rootboxes = rbi.get_rootboxes()
    for i, (example, target_label) in enumerate(zip(x, y)):
        linf_boxes = dist_to_boxes(example, rootboxes, active_map)
        l1_boxes = dist_to_boxes_l1(example, rootboxes, active_map)

        for idx in range(rootboxes.shape[0]):
            if linf_boxes[idx] > l_inf or l1_boxes[idx] > l_1:
                continue
            boxes = rbi.get_boxes_at_index(idx, target_label)
            dist_linf = dist_to_boxes(example, boxes, active_map)
            dist_l1 = dist_to_boxes_l1(example, boxes, active_map)

            if np.any((dist_linf < l_inf) & (dist_l1 < l_1)):
                sat += 1
                break


        if time.time() - t0 >= time_limit:
            break

    return {'sat': sat, 'n': len(x), 'time': time.time() - t0, 'index_time': index_time}

def run_robustness_task(at, oc_file, x, y, l_inf, l_1, time_limit, n):
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
    
    result = adv_robustness(
        at, oc_file, x_correct, y_correct, n, l_inf, l_1, time_limit=time_limit
    )

    return result

@njit
def dist_to_boxes_l1(example, boxes, feat_map):
    dist = np.zeros((boxes.shape[0],), dtype=np.float64)
    for s in range(boxes.shape[0]):
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
            x = example[i]
            lo, hi = los[idx], his[idx]

            d = 0.0
            t = lo - x
            if t > 0.0:
                d = t
            t = x - hi
            if t > d:
                d = t

            dist[s] += d
    return dist