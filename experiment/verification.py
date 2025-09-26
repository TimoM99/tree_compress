import time
import veritas

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


def approx_emp_robustness(at, example, target_label, max_delta):
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


def exact_emp_robustness(at, example, target_label, max_delta):
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


def emp_robustness(at, x, y, n, exact, timeout):
    delta_lo = 0.0
    count = 0

    if exact:
        f = exact_emp_robustness
    else:
        f = approx_emp_robustness

    t = time.time()
    for i in x.index:
        target_label = not (y.loc[i] > 0.0)
        example = x.loc[i, :].to_numpy()
        pred_label = at.eval(example)[0, 0] > 0.0

        if pred_label != target_label:
            res = f(at, example, target_label, max_delta=1.0)
            delta_lo += res

        count += 1
        if count >= n:
            break
        if time.time() - t > timeout:
            break
    t = time.time() - t

    return delta_lo / count, count, t


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

    ## VERIFICATION: (2) Empricial robustness (exact + approx)
    rob_exact, rob_exact_n, rob_exact_time = emp_robustness(
        at, x, y, n, exact=True, timeout=timeout
    )
    rob_approx, rob_approx_n, rob_approx_time = emp_robustness(
        at, x, y, n, exact=False, timeout=timeout
    )




    # isfair, fair_timeout, fair_time = fairness_task(at, timeout)

    return {
        # "nocs": nocs,
        # "nocs_time": nocs_time,
        # "nocs_timeout": nocs_timeout,
        "exact_emp_rob": rob_exact,
        "exact_emp_rob_n": rob_exact_n,
        "exact_emp_rob_timeout": rob_exact_time >= timeout,
        "exact_emp_rob_time": rob_exact_time,
        "approx_emp_rob": rob_approx,
        "approx_emp_rob_n": rob_approx_n,
        "approx_emp_rob_timeout": rob_approx_time >= timeout,
        "approx_emp_rob_time": rob_approx_time,
        # "isfair": isfair,
        # "fair_timeout": fair_timeout,
        # "fair_time": fair_time,
    }
