import numpy as np

def get_params_xgb(d):
    params = {
        "random_state": d.seed + 9348,
        "n_jobs": 1,
        "nthread": 1,
        "n_estimators": [10, 25, 50, 100],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.1, 0.25, 0.5, 1.0],
        "subsample": 1.0,
        "tree_method": "hist",
    }

    if d.is_multiclass():
        params["objective"] = "multi:softmax"
        params["num_class"] = d.num_classes
        #params["multi_strategy"] = "multi_output_tree"

    return params
    

# def get_params_xgb_optimal_dt(d):
#     params = {
#         "random_state": d.seed + 9348,
#         "n_jobs": 1,
#         "nthread": 1,
#         "n_estimators": [10],
#         "max_depth": [3],
#         #"max_leaves": [2**4, 2**6, 2**8],
#         "learning_rate": [0.1],
#         "subsample": [1.0],
#         #"colsample_bytree": [0.5, 0.75, 1.0],
#         "tree_method": "hist",
#     }

#     if d.is_multiclass():
#         params["objective"] = "multi:softmax"
#         params["num_class"] = d.num_classes
#         #params["multi_strategy"] = "multi_output_tree"

#     return params

def get_params_xgb_small(d, ntrees, nnodes, nnzleafs):
    n_estimators = np.unique([int(0.5 * ntrees), int(0.75 * ntrees), int(ntrees)])
    n_estimators = list(map(int, n_estimators))
    if 0 in n_estimators:
        n_estimators.remove(0)
    #max_leaves = range(2, int(np.ceil(np.log2(2 * nnodes / ntrees))))
    max_depth = range(2, int(np.ceil(np.log2(4 * nnodes / ntrees))))
    params = {
        "random_state": d.seed + 9348,
        "n_jobs": 1,
        "nthread": 1,
        "n_estimators": n_estimators,
        "max_depth": list(max_depth),
        #"max_leaves": list(map(lambda x: 2**x, max_depth)),
        "learning_rate": [1.0, 0.5, 0.25, 0.1],
        #"subsample": [0.5, 0.75, 1.0],
        #"colsample_bytree": [0.5, 0.75, 1.0],
        "tree_method": "hist",
    }

    if d.is_multiclass():
        params["objective"] = "multi:softmax"
        params["num_class"] = d.num_classes
        #params["multi_strategy"] = "multi_output_tree"

    return params

def get_params_lgb(d):
    params = {
        "random_state": d.seed + 9348,
        "verbosity": -1,
        "n_jobs": 1,
        "n_estimators": [50, 100],
        "num_leaves": [2**4, 2**8], #, 10]
        "learning_rate": [0.9, 0.5],#, 0.1],
        #"subsample": [0.5, 0.75, 1.0],
    }
    return params

def get_params_rf(d):
    params = {
        "random_state": d.seed + 9348,
        "n_jobs": 1,
        "n_estimators": [50, 100, 250],
        "max_depth": [10, 15],
        "max_leaf_nodes": [None],
    }
    return params


def get_params_dt(d):
    params = {
        "max_leaf_nodes": [None, 2048, 1024, 512, 128],
        "random_state": d.seed + 9348,
        "ccp_alpha": [0.0, 0.01, 0.1, 0.2],
    }
    if d.is_regression():
        params["criterion"] = ["absolute_error", "squared_error", "friedman_mse"]
    else:
        params["criterion"] = ["gini", "entropy", "log_loss"]
    return params

def get_params(d, model_type):
    if model_type == "rf":
        return get_params_rf(d)
    elif model_type == "xgb":
        return get_params_xgb(d)
    elif model_type == "lgb":
        return get_params_lgb(d)
    elif model_type == "dt":
        return get_params_dt(d)
    else:
        raise RuntimeError(f"get_params: model type {model_type}")

def get_params_specific(d, model_type, max_depth, n_estimators, lr):
    return {
        "random_state": d.seed + 9348,
        "n_jobs": 1,
        "nthread": 1,
        "n_estimators": [n_estimators],
        "max_depth": [max_depth],
        #"max_leaves": [2**4, 2**6, 2**8],
        "learning_rate": [lr],
        "subsample": [1.0],
        #"colsample_bytree": [0.5, 0.75, 1.0],
        "tree_method": "hist",
    }

def get_params_small(d, model_type, ntrees, nnodes, nnzleafs):
    if model_type == "rf":
        raise RuntimeError("not impl")
    elif model_type == "xgb":
        return get_params_xgb_small(d, ntrees, nnodes, nnzleafs)
    elif model_type == "lgb":
        raise RuntimeError("not impl")
    elif model_type == "dt":
        raise RuntimeError("not impl")
    else:
        raise RuntimeError(f"get_params: model type {model_type}")

def num_params(d, model_type):
    param_dict = get_params(d, model_type)
    return sum(1 for params in d.paramgrid(**param_dict))
