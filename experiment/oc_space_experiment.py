import click

import os
os.environ['PRADA_DATA_DIR']='/cw/dtaijupiter/NoCsBack/dtai/timo/prada_data'

from matplotlib.pylab import pareto
import veritas
import tree_compress
import numpy as np
import random
import json


import util
import model_params

from sklearn.metrics import balanced_accuracy_score
from verification import run_verification_tasks, count_ocs
import pickle

@click.group()
def cli():
    pass

@cli.command("count_ocs_small")
@click.argument("dname")
@click.option("--save", is_flag=True, default=False)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=21600)
@click.option("--fold", default=0)
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--abserr", default=0.005)
def compression_cmd(dname, save, seed, silent, timeout, fold, model_type, abserr):
    np.random.seed(seed)
    random.seed(seed)

    penalties = ['lop']
    
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)
    model_class = d.get_model_class(model_type)

    param_dict = model_params.get_params(d, model_type)
    for param in d.paramgrid(**param_dict):
        clf, train_time = dtrain.train(model_class, param)
        mtrain = dtrain.metric(clf)
        mvalid = dvalid.metric(clf)
        mtest =  dtest.metric(clf)
        at_orig = veritas.get_addtree(clf, silent=silent)
        n_leafs_before = at_orig.num_leafs()

        results = {
            'date_time': util.nowstr(),
            'hostname': os.uname()[1],
            'dname': dname,
            'model_type': model_type,
            'fold': fold,
            'seed': seed,
            'metric_name': d.metric_name,
            'train_time': train_time,
            'mtrain': mtrain,
            'mvalid': mvalid,
            'mtest': mtest,
            'ntrees': len(at_orig),
            'nnodes': int(at_orig.num_nodes()),
            'nleafs': n_leafs_before,
            'max_depth': int(at_orig.max_depth()),
            'params': param,
            'refinements': []
        }
        if save:
            results['model_json'] = at_orig.to_json()

        models = {}
        models['xgb'] = at_orig
        # Apply the different compression methods
        for penalty in penalties:
            if penalty == 'lop':
                data = tree_compress.Data(
                    dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                    dtest.X.to_numpy(), dtest.y.to_numpy(),
                    dvalid.X.to_numpy(), dvalid.y.to_numpy())
                
                compr = tree_compress.Compress(
                            data,
                            at_orig,
                            score=balanced_accuracy_score,
                            isworse=lambda v, ref: ref-v > abserr,
                            seed=5823,
                            silent=silent
                        )

                compr.no_convergence_warning = True
                at_refined = compr.compress(max_rounds=2, timeout=timeout)

            models[penalty] = at_refined

        for mtype, model in models.items():
            sum_depth = np.zeros(len(dtest.X))
            for t in model:
                nodes = t.eval_node(dtest.X)
                depths = [t.depth(node) for node in nodes]
                sum_depth += depths
            median_depth = np.median(sum_depth)
            
            results['refinements'].append({
                'penalty': mtype,
                'verification_results': run_verification_tasks(model, dtest.X, dtest.y, timeout=timeout, n=500),
                'nb_splits_done': median_depth,
                'memory_usage': 25*(model.num_nodes() + model.num_leafs()),
                'model_json': model.to_json() if save else None,
                'nleafs': model.num_leafs(),
                'nnodes': model.num_nodes(),
                'observed_oc_space_training': calculate_observable_oc_space(x=dtrain.X.to_numpy(), at=model),
                'observed_oc_space_test': calculate_observable_oc_space(x=dtest.X.to_numpy(), at=model),
                'oc_score_training': calculate_oc_score(x=dtrain.X.to_numpy(), y=dtrain.y.to_numpy(), at=model),
                'oc_score_test': calculate_oc_score(x=dtest.X.to_numpy(), y=dtest.y.to_numpy(), at=model),
                # 'oc_space': count_ocs(model, timeout=timeout),
                'oc_space_bound': bound_oc_space(model),
                'mtest': dtest.metric(model),
                'mvalid': dvalid.metric(model),
                'mtrain': dtrain.metric(model)
            })
        if not silent:
            __import__('pprint').pprint(results)
        print(json.dumps(results))


@cli.command("verify_compressed_models")
@click.argument("dname")
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=21600)
@click.option("--fold", default=0)
def verify_compressed_models_cmd(dname, seed, silent, timeout, fold):
    np.random.seed(seed)
    random.seed(seed)

    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)

    file = f"results/xgb_classification_saved.txt"
    with open(file, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if line_dict['dname'] != dname or line_dict['fold'] != fold:
                continue
            
            results = {
                'date_time': util.nowstr(),
                'hostname': os.uname()[1],
                'dname': dname,
                'fold': fold,
                'seed': seed,
                'metric_name': line_dict['metric_name'],
                'mtrain': line_dict['mtrain'],
                'mvalid': line_dict['mvalid'],
                'mtest': line_dict['mtest'],
                'ntrees': line_dict['ntrees'],
                'nnodes': line_dict['nnodes'],
                'nleafs': line_dict['nleafs'],
                'params': line_dict['params'],
            }

            model = veritas.AddTree.from_json(line_dict['refinements'][3]['model_json'])
            assert line_dict['refinements'][3]['penalty'] == 'ours'



            results['compression'] = {
                'verification_results': run_verification_tasks(model, dtest.X, dtest.y, timeout=timeout, n=500),
                'nleafs': model.num_leafs(),
                'nnodes': model.num_nodes(),
                'observed_oc_space_training': calculate_observable_oc_space(x=dtrain.X.to_numpy(), at=model),
                'observed_oc_space_test': calculate_observable_oc_space(x=dtest.X.to_numpy(), at=model),
                'oc_score_training': calculate_oc_score(x=dtrain.X.to_numpy(), y=dtrain.y.to_numpy(), at=model),
                'oc_score_test': calculate_oc_score(x=dtest.X.to_numpy(), y=dtest.y.to_numpy(), at=model),
                # 'oc_space': count_ocs(model, timeout=timeout),
                'oc_space_bound': bound_oc_space(model),
                'mtest': dtest.metric(model),
                'mvalid': dvalid.metric(model),
                'mtrain': dtrain.metric(model)
            }

            print(json.dumps(results))


@cli.command('verify_pareto_models_save')
@click.argument("dname")
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=86400)
@click.option("--memory_limit", default=64*1024*1024*1024)
@click.option("--fold", default=0)
@click.option("--fronts_file", default='pareto_fronts_LOP.pkl')
def verify_pareto_models_cmd(dname, seed, silent, timeout, memory_limit, fold, fronts_file):
    np.random.seed(seed)
    random.seed(seed)
    pareto_fronts_file = fronts_file

    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)

    # Load in results that we already have from linear scan
    results_linear_scan = {}
    with open('results/linear_scan_after_LOP.txt', 'r') as file: #Load in the linear scan results
        for line in file:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            key = f"{line_dict['dname']}_{line_dict['params']['n_estimators']}_{line_dict['params']['max_depth']}_{line_dict['params']['learning_rate']}_{line_dict['fold']}"
            results_linear_scan[key] = line_dict

    # Load in the pareto fronts
    with open(pareto_fronts_file, 'rb') as f:
        pareto_fronts = pickle.load(f)
        pareto_fronts = pareto_fronts[dname]

    # print(pareto_fronts)
    # Now go over all saved xgb models and verify those that are in the pareto front.
    # If we have verified them but failed, re-verify them and remove their current entry from linear_scan_after_LOP.txt
    file_xgb = f"results/xgb_classification_saved.txt"
    with open(file_xgb, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if line_dict['dname'] != dname or line_dict['fold'] != fold:
                continue
            # Check if this model is in the pareto front, if not we don't bother verifying it.
            if pareto_fronts[(pareto_fronts['n_estimators'].astype(float) == float(line_dict['params']['n_estimators'])) &
                        (pareto_fronts['max_depth'].astype(float) == float(line_dict['params']['max_depth'])) &
                        (pareto_fronts['learning_rate'].astype(float) == float(line_dict['params']['learning_rate']))]['on_front'].values[0] == False:
                continue
            else: # If model on pareto front, verify

                results = {
                    'date_time': util.nowstr(),
                    'hostname': os.uname()[1],
                    'dname': dname,
                    'fold': fold,
                    'seed': seed,
                    'metric_name': line_dict['metric_name'],
                    'mtrain': line_dict['mtrain'],
                    'mvalid': line_dict['mvalid'],
                    'mtest': line_dict['mtest'],
                    'ntrees': line_dict['ntrees'],
                    'nnodes': line_dict['nnodes'],
                    'nleafs': line_dict['nleafs'],
                    'params': line_dict['params'],
                }

                model = veritas.AddTree.from_json(line_dict['refinements'][3]['model_json'])
                assert line_dict['refinements'][3]['penalty'] == 'ours'

                verification_results, doms, inds = run_verification_tasks(model, dtest.X, dtest.y, timeout=timeout, memory_limit=memory_limit, n=500)

                np.save(f'/cw/dtailocal/timo/OCs/OC_boxes_positive_{dname}_{line_dict["params"]["n_estimators"]}_{line_dict["params"]["max_depth"]}_{line_dict["params"]["learning_rate"]}_fold{fold}', doms['positive'])
                np.save(f'/cw/dtailocal/timo/OCs//OC_boxes_negative_{dname}_{line_dict["params"]["n_estimators"]}_{line_dict["params"]["max_depth"]}_{line_dict["params"]["learning_rate"]}_fold{fold}', doms['negative'])
                np.save(f'/cw/dtailocal/timo/OCs/OC_feature_indices_positive_{dname}_{line_dict["params"]["n_estimators"]}_{line_dict["params"]["max_depth"]}_{line_dict["params"]["learning_rate"]}_fold{fold}', inds['positive'])
                np.save(f'/cw/dtailocal/timo/OCs/OC_feature_indices_negative_{dname}_{line_dict["params"]["n_estimators"]}_{line_dict["params"]["max_depth"]}_{line_dict["params"]["learning_rate"]}_fold{fold}', inds['negative'])

                results['compression'] = {
                    'verification_results': verification_results,
                    'nleafs': model.num_leafs(),
                    'nnodes': model.num_nodes(),
                    'observed_oc_space_training': calculate_observable_oc_space(x=dtrain.X.to_numpy(), at=model),
                    'observed_oc_space_test': calculate_observable_oc_space(x=dtest.X.to_numpy(), at=model),
                    'oc_score_training': calculate_oc_score(x=dtrain.X.to_numpy(), y=dtrain.y.to_numpy(), at=model),
                    'oc_score_test': calculate_oc_score(x=dtest.X.to_numpy(), y=dtest.y.to_numpy(), at=model),
                    'oc_space_bound': bound_oc_space(model),
                    'mtest': dtest.metric(model),
                    'mvalid': dvalid.metric(model),
                    'mtrain': dtrain.metric(model)
                }

                print(json.dumps(results))


@cli.command('verify_pareto_models')
@click.argument("dname")
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=86400)
@click.option("--memory_limit", default=64*1024*1024*1024)
@click.option("--fold", default=0)
@click.option("--fronts_file", default='pareto_fronts_LOP.pkl')
def verify_pareto_models_cmd(dname, seed, silent, timeout, memory_limit, fold, fronts_file):
    np.random.seed(seed)
    random.seed(seed)
    pareto_fronts_file = fronts_file

    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)

    # Load in results that we already have from linear scan
    results_linear_scan = {}
    with open('results/linear_scan_after_LOP.txt', 'r') as file: #Load in the linear scan results
        for line in file:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            key = f"{line_dict['dname']}_{line_dict['params']['n_estimators']}_{line_dict['params']['max_depth']}_{line_dict['params']['learning_rate']}_{line_dict['fold']}"
            results_linear_scan[key] = line_dict

    # Load in the pareto fronts
    with open(pareto_fronts_file, 'rb') as f:
        pareto_fronts = pickle.load(f)
        pareto_fronts = pareto_fronts[dname]

    # print(pareto_fronts)
    # Now go over all saved xgb models and verify those that are in the pareto front.
    # Check whether we haven't already verified them in the linear scan results. If so, skip them.
    # If we have verified them but failed, re-verify them and remove their current entry from linear_scan_after_LOP.txt
    file_xgb = f"results/xgb_classification_saved.txt"
    with open(file_xgb, "r") as f:
        for line in f:
            if not line.startswith('{'):
                continue
            line_dict = json.loads(line.strip())
            if line_dict['dname'] != dname or line_dict['fold'] != fold:
                continue
            # Check if this model is in the pareto front, if not we don't bother verifying it.
            if pareto_fronts[(pareto_fronts['n_estimators'].astype(float) == float(line_dict['params']['n_estimators'])) &
                        (pareto_fronts['max_depth'].astype(float) == float(line_dict['params']['max_depth'])) &
                        (pareto_fronts['learning_rate'].astype(float) == float(line_dict['params']['learning_rate']))]['on_front'].values[0] == False:
                continue
            else: # If model on pareto front, check if already verified in linear scan results
                key = f"{line_dict['dname']}_{line_dict['params']['n_estimators']}_{line_dict['params']['max_depth']}_{line_dict['params']['learning_rate']}_{line_dict['fold']}"
                if key in results_linear_scan and results_linear_scan[key]['compression']['verification_results']['failed_oc'] == False \
                    and 'time_taken_oc' in results_linear_scan[key]['compression']['verification_results']:
                        print(json.dumps(results_linear_scan[key]))
                else:
                    results = {
                        'date_time': util.nowstr(),
                        'hostname': os.uname()[1],
                        'dname': dname,
                        'fold': fold,
                        'seed': seed,
                        'metric_name': line_dict['metric_name'],
                        'mtrain': line_dict['mtrain'],
                        'mvalid': line_dict['mvalid'],
                        'mtest': line_dict['mtest'],
                        'ntrees': line_dict['ntrees'],
                        'nnodes': line_dict['nnodes'],
                        'nleafs': line_dict['nleafs'],
                        'params': line_dict['params'],
                    }

                    model = veritas.AddTree.from_json(line_dict['refinements'][3]['model_json'])
                    assert line_dict['refinements'][3]['penalty'] == 'ours'

                    results['compression'] = {
                        'verification_results': run_verification_tasks(model, dtest.X, dtest.y, timeout=timeout, memory_limit=memory_limit, n=500),
                        'nleafs': model.num_leafs(),
                        'nnodes': model.num_nodes(),
                        'observed_oc_space_training': calculate_observable_oc_space(x=dtrain.X.to_numpy(), at=model),
                        'observed_oc_space_test': calculate_observable_oc_space(x=dtest.X.to_numpy(), at=model),
                        'oc_score_training': calculate_oc_score(x=dtrain.X.to_numpy(), y=dtrain.y.to_numpy(), at=model),
                        'oc_score_test': calculate_oc_score(x=dtest.X.to_numpy(), y=dtest.y.to_numpy(), at=model),
                        'oc_space_bound': bound_oc_space(model),
                        'mtest': dtest.metric(model),
                        'mvalid': dvalid.metric(model),
                        'mtrain': dtrain.metric(model)
                    }

                    print(json.dumps(results))




def calculate_oc_score(x, y, at):
    """
    Calculates the oc-score per class.
    Returns a dict: {class_label: oc_score}
    """
    n_datapoints = x.shape[0]
    n_trees = len(at)
    configuration = np.zeros((n_datapoints, n_trees), dtype=np.int32)
    for i, t in enumerate(at):
        configuration[:, i] = t.eval_node(x)

    classes = np.unique(y)
    oc_scores_per_class = {}

    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) <= 1:
            oc_scores_per_class[cls] = np.nan
            continue

        config_cls = configuration[idx]
        oc_space = np.unique(config_cls, axis=0)

        if len(oc_space) <= 1:
            oc_scores_per_class[cls] = np.nan
            continue

        oc_space_tuples = {tuple(row): i for i, row in enumerate(oc_space)}
        config_indices = np.array([oc_space_tuples[tuple(row)] for row in config_cls])

        diffs = oc_space[:, None, :] != oc_space[None, :, :]
        hamming_matrix = np.sum(diffs, axis=2).astype(float)
        np.fill_diagonal(hamming_matrix, np.inf)

        ks = [1, 10, 100, 1000]  # You can adjust or parameterize these k values as needed
        oc_scores_dict = {}
        for k in ks:
            if hamming_matrix.shape[1] > k:
                oc_scores = np.sort(hamming_matrix[config_indices], axis=1)[:, k]
                oc_scores_dict[f"oc_score_k{k}"] = np.mean(oc_scores)
            else:
                oc_scores_dict[f"oc_score_k{k}"] = np.nan
        oc_scores_per_class[str(cls)] = oc_scores_dict

    return oc_scores_per_class

def bound_oc_space(at):
    splits = at.get_splits()
    bound_1 = 1
    for _, f_values in splits.items():
        bound_1 *= (len(f_values) + 1)

    bound_2 = 1
    for t in at:
        bound_2 *= (t.num_leaves())
    return bound_1, bound_2

def calculate_observable_oc_space(x, at):
        # Find the leaf values
        oc_space = set()

        # For each datapoint, collect the leaf id from each tree
        n_datapoints = x.shape[0]
        n_trees = len(at)
        # Each row: one datapoint, each column: leaf id from one tree
        configuration = np.zeros((n_datapoints, n_trees), dtype=int)
        for i, t in enumerate(at):
            configuration[:, i] = t.eval_node(x)
        # print(configuration)
        oc_space.update(map(tuple, configuration))
        return len(oc_space)



if __name__ == "__main__":
    cli()