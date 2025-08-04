from turtle import pen
import click
from matplotlib.pylab import ifft
import torch
import torch.nn as nn
import os

os.environ['PRADA_DATA_DIR']='/cw/dtaijupiter/NoCsBack/dtai/timo/prada_data'
import json
import random
import ast

import veritas
import tree_compress
import numpy as np
import time
import warnings
import prada

import util
import sys
from tree_compress.forestprune_original import difference_array_list, evaluate, get_node_count, nodes_per_layer, prune_polish, solve_weighted, total_nodes
import model_params
from verification import run_verification_tasks

from scipy.sparse import csr_matrix
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error
from sklearn.svm import LinearSVC, LinearSVR


@click.group()
def cli():
    pass

@cli.command("list_compression_classification")
@click.option("--save", is_flag=True, default=False)
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--abserr", default=0.005)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
def print_configs(save, model_type, abserr, seed, silent):
    for dname in util.DNAMES_SUBSUB:
        d = prada.get_dataset(dname, seed=seed, silent=True)

        folds = [i for i in range(util.NFOLDS)]

        grid = d.paramgrid(fold=folds)

        for cli_param in grid:
            print("python3 compression_experiment.py compression_classification",
                  dname,
                  "--save" if save else "",
                  "--model_type", model_type,
                  "--fold", cli_param["fold"],
                  "--abserr", abserr,
                  "--seed", seed,
                  "--silent" if silent else "")



@cli.command("compression_classification")
@click.argument("dname")
@click.option("--save", is_flag=True, default=False)
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.005)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=21600)
def compression_cmd(dname, save, model_type, fold, abserr, seed, silent, timeout):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    penalties = ['gr', 'ic', 'lrl1', 'ours', 'forestprune']
    
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)
    model_class = d.get_model_class(model_type)

    param_dict = model_params.get_params(d, model_type)
    for param in d.paramgrid(**param_dict):
        # Fit xgb model
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

        # Apply the different compression methods
        for penalty in penalties:
            refine_time = time.time()

            sparse_train_x = transform_data_sparse(at_orig, dtrain.X)
            sparse_valid_x = transform_data_sparse(at_orig, dvalid.X)

            if penalty == "lrl1":
                timer = time.time()
                accuratest = mvalid
                best_alpha = 0.0
                smallest = n_leafs_before
                alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.925,0.955,0.975,1]
                for alpha in alpha_list:
                    if time.time() - timer > timeout:
                        break
                    refiner = LRPlusL1Refiner()
                    refiner.set_params(at_orig, alpha)
                    np.random.seed(seed)
                    refiner.refine(50, sparse_train_x, dtrain.y)

                    preds = refiner.forward_sparse(sparse_valid_x)
                    score = dvalid.metric(preds > 0.5)

                    # Note: In case of random forests, predictions are calculated by taking the 
                    # average prediction of all trees. However, if we change the leaf values using 
                    # this refiner, the predictions of a tree no longer represent a probability, but 
                    # a logit. Positive logits -> positive prediction, negative logits -> negative prediction.
                    # The RF AddTree will thus calculate the average logit instead of adding them. This makes
                    # that the probabilities are not correct! But, since we only care about balanced accuracy,
                    # it's the target prediction that matters and this remains the same.
                    at_compr = at_orig.copy()
                    at_compr = set_new_addtree(
                        refiner.tree_weights.detach().numpy(),
                        [w.detach().numpy() for w in refiner.leaf_weights],
                        refiner.base_score.detach().numpy(),
                        at_compr)
                    
                    num_leafs_after = at_compr.num_leafs()
                    
                    if score > mvalid - abserr:
                        if (num_leafs_after < smallest):
                            smallest = num_leafs_after
                            best_alpha = alpha
                            accuratest = score
                        elif (smallest == num_leafs_after) and (accuratest < score):
                            best_alpha = alpha
                            accuratest = score

                at_refined = at_orig.copy()
                # If the best alpha is 0.0, then we don't need to refine the tree.
                # because we couldn't find a better tree or a timeout was reached.
                if best_alpha != 0.0:
    
                    refiner.set_params(at_orig, best_alpha)
                    np.random.seed(seed)
                    refiner.refine(50, sparse_train_x, dtrain.y)
                    
                    at_refined = set_new_addtree(
                        refiner.tree_weights.detach().numpy(),
                        [w.detach().numpy() for w in refiner.leaf_weights],
                        refiner.base_score.detach().numpy(),
                        at_refined)

            elif penalty == "ic":
                # Calculate the prediction probability for each tree in the ensemble.
                ensemble_proba = np.asarray([sigmoid(t.eval(dtrain.X)) if at_orig.get_type() == veritas.AddTreeType.CLF_SOFTMAX else t.eval(dtrain.X) for t in at_orig])
                ensemble_proba = np.concatenate((1 - ensemble_proba, ensemble_proba), axis=2)
                ic_list = individual_contribution(ensemble_proba, dtrain.y.to_numpy())
                # Individual contribution is negative, so we sort in ascending order.
                sorted_trees = [at_orig[i] for i in np.argsort(ic_list)]
                at_refined = veritas.AddTree(at_orig.num_leaf_values(), at_orig.get_type())
                at_refined.set_base_score(0, at_orig.get_base_score(0) if at_orig.get_type() == veritas.AddTreeType.CLF_SOFTMAX else 0)

                # Keep adding trees until the validation score is within the error margin.
                score = -np.inf
                nb_trees = 0
                while (score < mvalid - abserr) and (nb_trees < len(sorted_trees)):
                    at_refined.add_tree(sorted_trees[nb_trees])
                    if at_refined.get_type() == veritas.AddTreeType.CLF_MEAN:
                        at_refined.set_base_score(0, -len(at_refined)/2)
                    score = dvalid.metric(at_refined)
                    nb_trees += 1


            elif penalty == "ours":
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
                best_alpha = compr.records[-1].alpha
            
            elif penalty == 'gr':
                
                timer = time.time()
                C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1] # Values as suggested by paper.
                improvement = True
                refiner = LinearSVC( #The paper optimizes a linear SVM with hinge loss and L2 regularisation to optimize leaf values.
                    penalty='l2',
                    loss='hinge',
                    dual=True,
                    max_iter=1000,
                    fit_intercept=True,
                    random_state=3582134
                )
                # Initially, the refined at is the same as the original at.
                at_refined = at_orig.copy()
                while improvement:
                    if time.time() - timer > timeout:
                        break
                    best_c = None
                    best_score = -np.inf
                    for c in C:
                        refiner.set_params(C=c)
                        # Optimize W
                        sparse_train_x = transform_data_sparse(at_refined, dtrain.X)
                        refiner.fit(sparse_train_x, dtrain.y)
                        at_optimized = at_refined.copy()
                        at_optimized = set_new_leaf_vals(at_optimized, refiner.intercept_[0], refiner.coef_[0])
                        # Check if optimized tree score (with the current regularization strength) is better than best_score.
                        score_temp = dvalid.metric(at_optimized)
                        if score_temp > best_score:
                            best_score = score_temp
                            best_c = c

                    # Using best C value to optimize W
                    refiner.set_params(C=best_c)
                    refiner.fit(sparse_train_x, dtrain.y)
                    at_optimized = at_refined.copy()
                    at_optimized = set_new_leaf_vals(at_optimized, refiner.intercept_[0], refiner.coef_[0])
                    # Find difference threshold for merging neighbouring leaves
                    threshold = find_pruning_threshold(at_optimized, 10) # The paper suggests to prune 10% per iteration
                    # Merge neighbouring leaves
                    at_pruned = gr_prune(at_optimized, threshold)
                    # Check if after pruning we don't exceed max abserr
                    if (dvalid.metric(at_orig) - dvalid.metric(at_pruned) > abserr) \
                        or (at_pruned.num_leafs() == at_optimized.num_leafs()) \
                        or (at_pruned.num_leafs() == len(at_pruned)): # In the extreme case that all leaves are pruned, stop the algorithm -> example bacc(at_orig) < 0.500 + abs_err, then the pruned tree will be 0.5.
                        improvement = False
                    else:
                        at_refined = at_pruned
                
            elif penalty == "forestprune":
                timer = time.time()
                data = tree_compress.Data(
                    dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                    dtest.X.to_numpy(), dtest.y.to_numpy(),
                    dvalid.X.to_numpy(), dvalid.y.to_numpy())
                
                fprune = tree_compress.ForestPrune(data, clf, seed=util.SEED)
                time_spent = time.time() - timer
                at_refined = fprune.prune(timeout - time_spent)

            else:
                raise RuntimeError(f"{penalty} refiner has not been implemented yet.")

            refine_time = time.time() - refine_time
            mtrain_refined = dtrain.metric(at_refined)
            mvalid_refined = dvalid.metric(at_refined)
            mtest_refined = dtest.metric(at_refined)
            
            
            results['refinements'].append({
                'penalty': penalty,
                'compr_time': refine_time,
                'mtrain': mtrain_refined,
                'mvalid': mvalid_refined,
                'mtest': mtest_refined,
                'ntrees': len(at_refined),
                'nnodes': int(at_refined.num_nodes()),
                'nleafs': at_refined.num_leafs(),
                'max_depth': int(at_refined.max_depth()),
            })
            
            if penalty == 'ours':
                results['refinements'][-1]['best_alpha'] = best_alpha
            if save:
                results['refinements'][-1]['model_json'] = at_refined.to_json()
            
        if not silent:
            __import__('pprint').pprint(results)
        print(json.dumps(results))



@cli.command("list_compression_regression")
@click.option("--save", is_flag=True, default=False)
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--abserr", default=0.02)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
@click.option("--timeout", default=21600)
def print_configs(save, model_type, abserr, seed, silent, timeout):
    for dname in util.DNAMES_REGRESSION:
        d = prada.get_dataset(dname, seed=seed, silent=True)

        folds = [i for i in range(util.NFOLDS)]

        grid = d.paramgrid(fold=folds)

        for cli_param in grid:
            print("python3 compression_experiment.py compression_regression",
                  dname,
                  "--save" if save else "",
                  "--model_type", model_type,
                  "--fold", cli_param["fold"],
                  "--abserr", abserr,
                  "--seed", seed,
                  "--silent" if silent else "")

@cli.command("compression_regression")
@click.argument("dname")
@click.option("--save", is_flag=True, default=False)
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.02)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=21600)
def compression_regression_cmd(dname, save, model_type, fold, abserr, seed, silent, timeout):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    penalties = ['gr', 'lrl1', 'ours', 'forestprune']
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)
    model_class = d.get_model_class(model_type)

    
    param_dict = model_params.get_params(d, model_type)
    for param in d.paramgrid(**param_dict):
        # Fit xgb model
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
        
        # Apply the different compression methods
        for penalty in penalties:
            refine_time = time.time()

            sparse_train_x = transform_data_sparse(at_orig, dtrain.X)
            sparse_valid_x = transform_data_sparse(at_orig, dvalid.X)
            

            if penalty == "lrl1":
                at_reg = at_orig.copy()
                # Convert bagging ensemble to boosted ensemble, because the way that bagging ensemble
                # makes predictions is not compatible with how the LRPlusL1Refiner does.
                if at_orig.get_type() == veritas.AddTreeType.REGR_MEAN:
                    at_reg = transform_to_regular_regr(at_reg)

                timer = time.time()
                accuratest = mvalid
                best_alpha = 0.0
                smallest = n_leafs_before
                alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.925,0.955,0.975,1]
                for alpha in alpha_list:
                    if time.time() - timer > timeout:
                        break
                    refiner = LRPlusL1Refiner()
                    # Set refiner to use regression mode -> avoids sigmoid function for probabilities
                    refiner.set_params(at_reg, alpha, regression=True)
                    # We have to set the seed such that the refiner is deterministic when we select alpha.
                    np.random.seed(seed)
                    refiner.refine(50, sparse_train_x, dtrain.y)

                    # If sparse_valid_x is too big, adapt forward_sparse method to work for regression.
                    preds = refiner(torch.from_numpy(sparse_valid_x.todense())).detach().numpy()
                    score = dvalid.metric(preds)
                    # First copy the tree, otherwise set_new_addtree will change the original tree.
                    at_compr = at_reg.copy()
                    at_compr = set_new_addtree(
                        refiner.tree_weights.detach().numpy(),
                        [w.detach().numpy() for w in refiner.leaf_weights],
                        refiner.base_score.detach().numpy(),
                        at_compr)
                    num_leafs_after = at_compr.num_leafs()
                    
                    # Relative error for rmse
                    if score < mvalid * (1 + abserr):
                        if (num_leafs_after < smallest):
                            smallest = num_leafs_after
                            best_alpha = alpha
                            accuratest = score
                        elif (smallest == num_leafs_after) and (accuratest > score):
                            best_alpha = alpha
                            accuratest = score

                at_refined = at_reg.copy()
                if best_alpha != 0.0:
                    refiner.set_params(at_reg, best_alpha, regression=True)
                    np.random.seed(seed)
                    refiner.refine(50, sparse_train_x, dtrain.y)
                    at_refined = set_new_addtree(
                        refiner.tree_weights.detach().numpy(),
                        [w.detach().numpy() for w in refiner.leaf_weights],
                        refiner.base_score.detach().numpy(),
                        at_refined)

            elif penalty == "ours":
                data = tree_compress.Data(
                    dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                    dtest.X.to_numpy(), dtest.y.to_numpy(),
                    dvalid.X.to_numpy(), dvalid.y.to_numpy())
                compr = tree_compress.Compress(
                            data,
                            at_orig,
                            score=root_mean_squared_error,
                            # Use a relative error for rmse as it differs massively per dataset
                            isworse=lambda v, ref: ref*(1 + abserr) < v,
                            seed=5823,
                            silent=True
                        )

                compr.no_convergence_warning = True
                at_refined = compr.compress(max_rounds=2, timeout=timeout)
                best_alpha = compr.records[-1].alphas
            
            # Below can be uncommented to check performance of the original forestprune code without transforming to AddTree.
            # elif penalty == "fp_orig":
            #     import warnings
            #     warnings.filterwarnings("ignore")

            #     tree_list = np.array(clf.estimators_)
            #     W_array = nodes_per_layer(tree_list)
            #     normalization = total_nodes(tree_list)
            #     learning_rate = 1/len(tree_list)

            #     base_err = dtrain.metric(at_orig)
            #     val_err = dvalid.metric(at_orig)

            #     print(val_err)
            #     print(dvalid.metric(clf.predict(dvalid.X)))

            #     diff_array_list = difference_array_list(dtrain.X, tree_list)
            #     diff_test_array_list = difference_array_list(dtest.X, tree_list)
            #     diff_val_array_list = difference_array_list(dvalid.X, tree_list)

            #     results_ = []
            #     warm_start = []
                
            #     print(dvalid.metric(evaluate(diff_val_array_list, dvalid.y, warm_start, [], learning_rate)))
            #     for alpha in np.flip(np.logspace(-6, 2.5, 100)):
            #         np.random.seed(seed)
            #         vars1, iters = solve_weighted(dtrain.y, tree_list, diff_array_list, alpha, learning_rate, W_array, normalization, warm_start=warm_start)
            #         warm_start = vars1
            #         coef = prune_polish(diff_array_list, dtrain.y, vars1, learning_rate)

            #         pred = evaluate(diff_val_array_list, dvalid.y, vars1, coef, learning_rate)
            #         val_err_new = dvalid.metric(pred)
            #         if val_err_new < val_err * (1 + abserr):
            #             results_.append((alpha, val_err_new, vars1, coef, iters))

            #         print(val_err_new)

            #     if len(results_) > 0:
            #         # Sort results by validation error
            #         results_.sort(key=lambda x: x[1])
            #         best_alpha, mvalid_refined, vars1, coef, iters = results_[0]
            #         pred = evaluate(diff_test_array_list, dtest.y, vars1, coef, learning_rate)
            #         mtest_refined = dtest.metric(pred)
            #         # mtrain_refined = dtrain.metric(pred)
            #         pruned_nodes = get_node_count(tree_list, vars1)
                
            #         print('mvalid_refined', mvalid_refined)
            #         print('mtest_refined', mtest_refined)
            #         # print('mtrain_refined', mtrain_refined)
            #         print('pruned_nodes', pruned_nodes)

            #     at_refined = at_orig.copy()


            elif penalty == 'gr':
                at_refined = at_orig.copy()
                # Same thing as for LRL1, convert bagging ensemble to boosted ensemble, because the way that bagging ensemble
                # makes predictions is not compatible with how the GR refiner does.
                if at_refined.get_type() == veritas.AddTreeType.REGR_MEAN:
                    at_refined = transform_to_regular_regr(at_refined)
                
                timer = time.time()
                C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1] # Values as suggested by paper.
                improvement = True
                refiner = LinearSVR( #The paper optimizes a mean squared error for regression ensembles
                    loss='squared_epsilon_insensitive',
                    dual=False,
                    max_iter=1000,
                    fit_intercept=True,
                    random_state=3582134
                )

                while improvement:
                    if time.time() - timer > timeout:
                        break
                    best_c = None
                    best_score = np.inf
                    for c in C:
                    
                        refiner.set_params(C=c)
                        
                        sparse_train_x = transform_data_sparse(at_refined, dtrain.X)
                        refiner.fit(sparse_train_x, dtrain.y)
                        
                        at_optimized = at_refined.copy()
                        at_optimized = set_new_leaf_vals(at_optimized, refiner.intercept_[0], refiner.coef_)
                        # Check if optimized tree score (with the current regularization strength) is better than best_score.
                        score_temp = dvalid.metric(at_optimized)
                        if score_temp < best_score:
                            best_score = score_temp
                            best_c = c

                    # Using best C value to optimize W
                    refiner.set_params(C=best_c)
                    refiner.fit(sparse_train_x, dtrain.y)
                    at_optimized = at_refined.copy()
                    at_optimized = set_new_leaf_vals(at_optimized, refiner.intercept_[0], refiner.coef_)
                    # Find difference threshold for merging neighbouring leaves
                    threshold = find_pruning_threshold(at_optimized, 10) # The paper suggests to prune 10% per iteration
                    # Merge neighbouring leaves
                    at_pruned = gr_prune(at_optimized, threshold)
        
                    # Check if after pruning we don't exceed max abserr
                    if (dvalid.metric(at_orig) * (1 + abserr) < dvalid.metric(at_pruned)) \
                        or (at_pruned.num_leafs() == at_optimized.num_leafs()) \
                        or (at_pruned.num_leafs() == len(at_pruned)): # In the extreme case that all leaves are pruned, stop the algorithm -> example bacc(at_orig) < 0.500 + abs_err, then the pruned tree will be 0.5.
                        improvement = False
                    else:
                        at_refined = at_pruned
                
            elif penalty == "forestprune":
                timer = time.time()
                data = tree_compress.Data(
                    dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                    dtest.X.to_numpy(), dtest.y.to_numpy(),
                    dvalid.X.to_numpy(), dvalid.y.to_numpy())

                fprune = tree_compress.ForestPrune(data, clf, max_mvalid_drop=abserr, seed=util.SEED)
                time_spent = time.time() - timer
                at_refined = fprune.prune(timeout - time_spent)
            else:
                raise RuntimeError(f"{penalty} refiner has not been implemented yet.")

            refine_time = time.time() - refine_time
            mtrain_refined = dtrain.metric(at_refined)
            mvalid_refined = dvalid.metric(at_refined)
            mtest_refined = dtest.metric(at_refined)
            
            
            results['refinements'].append({
                'penalty': penalty,
                'compr_time': refine_time,
                'mtrain': mtrain_refined,
                'mvalid': mvalid_refined,
                'mtest': mtest_refined,
                'ntrees': len(at_refined),
                'nnodes': int(at_refined.num_nodes()),
                'nleafs': at_refined.num_leafs(),
                'max_depth': int(at_refined.max_depth()),
            })
            if penalty == 'ours':
                results['refinements'][-1]['best_alpha'] = best_alpha
            if save:
                results['refinements'][-1]['model_json'] = at_refined.to_json()
            
        if not silent:
            __import__('pprint').pprint(results)
        print(json.dumps(results))



@cli.command("list_sensitivity_analysis")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
def print_configs(model_type, seed, silent):
    for dname in util.DNAMES_SUBSUBSUB:
        d = prada.get_dataset(dname, seed=seed, silent=True)

        folds = [i for i in range(util.NFOLDS)]
        abserrs = [0.0025, 0.005, 0.01, 0.02]
        grid = d.paramgrid(fold=folds, abserr=abserrs)

        for cli_param in grid:
            print("python3 compression_experiment.py sensitivity_analysis",
                  dname,
                  "--model_type", model_type,
                  "--fold", cli_param["fold"],
                  "--abserr", cli_param["abserr"],
                  "--seed", seed,
                  "--silent" if silent else "")
            

@cli.command("sensitivity_analysis")
@click.argument("dname")
@click.option("--save", is_flag=True, default=False)
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--fold", default=0)
@click.option("--abserr")
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
def sensitivity_cmd(dname, save, model_type, fold, abserr, seed, silent):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)
    model_class = d.get_model_class(model_type)

    param_dict = model_params.get_params(d, model_type)
    for param in d.paramgrid(**param_dict):
        # Fit xgb model
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
            'abserr': abserr,
            'compressions': []
        }
        
        for nb_rounds in [1, 2, 3]:
            refine_time = time.time()

            data = tree_compress.Data(
                dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                dtest.X.to_numpy(), dtest.y.to_numpy(),
                dvalid.X.to_numpy(), dvalid.y.to_numpy())

            compr = tree_compress.Compress(
                        data,
                        at_orig,
                        score=balanced_accuracy_score,
                        isworse=lambda v, ref: ref-v > float(abserr),
                        seed=5823,
                        silent=silent
                    )

            compr.no_convergence_warning = True
            at_refined = compr.compress(max_rounds=nb_rounds)
            best_alpha = compr.records[-1].alpha

            refine_time = time.time() - refine_time
            mtrain_refined = dtrain.metric(at_refined)
            mvalid_refined = dvalid.metric(at_refined)
            mtest_refined = dtest.metric(at_refined)
            
            results['compressions'].append({
                'max_rounds': nb_rounds,
                'compr_time': refine_time,
                'mtrain': mtrain_refined,
                'mvalid': mvalid_refined,
                'mtest': mtest_refined,
                'ntrees': len(at_refined),
                'nnodes': int(at_refined.num_nodes()),
                'nleafs': at_refined.num_leafs(),
                'max_depth': int(at_refined.max_depth()),
            })
            results['compressions'][-1]['best_alpha'] = best_alpha
            if save:
                results['compressions'][-1]['model_json'] = at_refined.to_json()
            
        if not silent:
            __import__('pprint').pprint(results)
        print(json.dumps(results))

@cli.command("list_practical_metrics")
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
@click.option("--file", default='results/xgb_classification_saved.txt')
def print_configs(seed, silent, file):
    for dname in util.DNAMES_SUBSUB:
        d = prada.get_dataset(dname, seed=seed, silent=True)

        folds = [i for i in range(util.NFOLDS)]
        n_estimators = [10, 25, 50, 100]
        max_depth = [4, 6, 8]
        lr = [1.0, 0.5, 0.25, 0.1]
        grid = d.paramgrid(fold=folds, n_estimators=n_estimators, max_depth=max_depth, lr=lr)


        for cli_param in grid:
            print("python3 compression_experiment.py practical_metrics",
                  dname,
                  "--depth", cli_param["max_depth"],
                  "--n_estimators", cli_param["n_estimators"],
                  "--lr", cli_param["lr"],
                  "--file", file,
                  "--fold", cli_param["fold"],
                  "--seed", seed,
                  "--silent" if silent else "")

@cli.command("practical_metrics")
@click.argument("dname")
@click.option("--depth", default=4)
@click.option("--n_estimators", default=10)
@click.option("--lr", default=0.1)
@click.option("--file", default='results/xgb_classification_saved.txt')
@click.option("--fold", default=0)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
def verification_cmd(dname, depth, n_estimators, lr, file, fold, seed, silent):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    data = {}
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('{'):
                continue
            line_dict = ast.literal_eval(line)
            key = f"{line_dict['dname']}_{line_dict['params']['n_estimators']}_{line_dict['params']['max_depth']}_{line_dict['params']['learning_rate']}_{line_dict['fold']}"
            data[key] = line_dict


    models = {}
    key = f"{dname}_{n_estimators}_{depth}_{lr}_{fold}"
    data = data[key]
    models['xgb'] = veritas.AddTree.from_json(data['model_json'])
    for refinement in data['refinements']:
        models[refinement['penalty']] = veritas.AddTree.from_json(refinement['model_json'])

    _, _, _, dtest = util.get_dataset(dname, seed, fold, silent)
    
    results = {
        'date_time': util.nowstr(),
        'hostname': os.uname()[1],
        'dname': dname,
        'fold': fold,
        'seed': seed,
        'depth': depth,
        'n_estimators': n_estimators,
        'lr': lr,
        'verifications': [],
        'nb_splits': [],
        'memory': []
    }
    
    for mtype, model in models.items():
        results['verifications'].append({
            'penalty': mtype,
            'verification_results': run_verification_tasks(model, dtest.X, dtest.y, timeout=1800, n=500)
        })

        sum_depth = np.zeros(len(dtest.X))
        for t in model:
            nodes = t.eval_node(dtest.X)
            depths = [t.depth(node) for node in nodes]
            sum_depth += depths
        median_depth = np.median(sum_depth)
        
        results['nb_splits'].append({
            'penalty': mtype,
            'nb_splits_done': median_depth
        })

        results['memory'].append({
            'penalty': mtype,
            'memory_usage': 25*(model.num_nodes() + model.num_leafs())
        })
    if not silent:
        __import__('pprint').pprint(results)
    print(json.dumps(results))
    
@cli.command("verification_oc")
@click.argument("dname")
@click.option("--save", is_flag=True, default=False)
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.02)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=True)
@click.option("--timeout", default=21600)
def compression_cmd(dname, save, model_type, fold, abserr, seed, silent, timeout):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    penalties = ['lop', 'lop-oc']
    
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, fold, silent)
    model_class = d.get_model_class(model_type)

    param_dict = model_params.get_params(d, model_type)
    for param in d.paramgrid(**param_dict):
        # Fit xgb model
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
            
            if penalty == 'lop-oc':
                data = tree_compress.Data(
                    dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                    dtest.X.to_numpy(), dtest.y.to_numpy(),
                    dvalid.X.to_numpy(), dvalid.y.to_numpy())
                
                compr = tree_compress.oc_compress.Compress(
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
                'verification_results': run_verification_tasks(model, dtest.X, dtest.y, timeout=1800, n=500),
                'nb_splits_done': median_depth,
                'memory_usage': 25*(model.num_nodes() + model.num_leafs()),
                'model_json': model.to_json() if save else None
            })
        if not silent:
            __import__('pprint').pprint(results)
        print(json.dumps(results))


def transform_to_regular_regr(at):
    at_result = veritas.AddTree(1, veritas.AddTreeType.REGR)
    at_result.set_base_score(0, at.get_base_score(0))
    nb_trees = len(at)
    for t in at:
        for leaf in t.get_leaf_ids():
            t.set_leaf_value(leaf, t.get_leaf_value(leaf, 0)/nb_trees)
        at_result.add_tree(t)
    return at_result



def find_pruning_threshold(at, pct):
    diff = []
    for tree_index, t in enumerate(at):
        if t.is_leaf(t.root()):
            continue
        else:
            nodes = [t.root()]
            while len(nodes) > 0:
                node = nodes.pop()
                if t.is_leaf(t.left(node)) and t.is_leaf(t.right(node)):
                    llv = t.get_leaf_value(t.left(node), 0)
                    rlv = t.get_leaf_value(t.right(node), 0)
                    diff.append(np.sum((llv - rlv)**2))
                else:
                    if not t.is_leaf(t.left(node)):
                        nodes.append(t.left(node))
                    if not t.is_leaf(t.right(node)):
                        nodes.append(t.right(node))
    return np.percentile(diff, pct)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gr_prune(at, threshold):
    pruned_at = veritas.AddTree(at.num_leaf_values(), at.get_type())
    pruned_at.set_base_score(0, at.get_base_score(0))
    for tree_index, t in enumerate(at):
        pruned_t = pruned_at.add_tree()
        nodes = [(t.root(), t.root())]
        while len(nodes) > 0:
            node, pruned_node = nodes.pop(0)
            if t.is_leaf(node):
                pruned_t.set_leaf_value(pruned_node, 0, t.get_leaf_value(node, 0))
            else:
                if t.is_leaf(t.left(node)) and t.is_leaf(t.right(node)):
                    llv = t.get_leaf_value(t.left(node), 0)
                    rlv = t.get_leaf_value(t.right(node), 0)
                    # Has to be <= because otherwise you can be stuck in an infinite loop by not pruning anything.
                    if np.sum((llv - rlv)**2) <= threshold:
                        pruned_t.set_leaf_value(pruned_node, 0, (llv + rlv)/2)
                    else:
                        split = t.get_split(node)
                        pruned_t.split(pruned_node, split.feat_id, split.split_value)
                        nodes = [(t.left(node), pruned_t.left(pruned_node)),
                                 (t.right(node), pruned_t.right(pruned_node))] + nodes
                else:
                    split = t.get_split(node)
                    pruned_t.split(pruned_node, split.feat_id, split.split_value)
                    nodes = [(t.left(node), pruned_t.left(pruned_node)),
                                 (t.right(node), pruned_t.right(pruned_node))] + nodes
    return pruned_at

def transform_data_sparse(at, x):
    row_ind, col_ind = [], []
    num_rows = x.shape[0]
    num_cols = 0

    for tree_index, t in enumerate(at):
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)
        offset = num_cols
        num_cols += num_leaves

        leaf2index = np.zeros(t.num_nodes(), dtype=int)
        for cnt, id in enumerate(leaf_ids):
            leaf2index[id] = cnt

        row_ind += list(range(num_rows))
        col_ind += list(map(lambda u: offset + leaf2index[u], t.eval_node(x)))

    return csr_matrix((np.ones(len(row_ind), dtype=np.float64),
                              (row_ind, col_ind)), shape=(num_rows, num_cols))


def set_new_leaf_vals(at, base_score, new_leaf_vals):
    at.set_base_score(0, base_score)
    offset = 0
    for tree_index, t in enumerate(at):
        for count , leaf_id in enumerate(t.get_leaf_ids()):
            t.set_leaf_value(leaf_id, 0, new_leaf_vals[offset + count])
        offset += count + 1
    return at


def set_new_addtree(tree_weights, leaf_weights, base_score, at):
    new_at = veritas.AddTree(at.num_leaf_values(), at.get_type())
    new_at.set_base_score(0, base_score)
    for tree_index, t in enumerate(at):
        if tree_weights[tree_index] != 0:
            for count, leaf_id in enumerate(t.get_leaf_ids()):
                t.set_leaf_value(leaf_id, 0, leaf_weights[tree_index][count] * tree_weights[tree_index])
            new_at.add_tree(t)
    return new_at

def individual_contribution(ensemble_proba, target):
    """
    Compute the individual contributions of each classifier wrt. the entire ensemble. Return the negative contribution due to the minimization.
    Source: https://github.com/sbuschjaeger/PyPruning/blob/master/PyPruning/RankPruningClassifier.py
    Reference:
        Lu, Z., Wu, X., Zhu, X., & Bongard, J. (2010). Ensemble pruning via individual contribution ordering. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 871–880. https://doi.org/10.1145/1835804.1835914
    """

    ic_list = []
    for i in range(len(ensemble_proba)):
        iproba = ensemble_proba[i,:,:]
        n = iproba.shape[0]

        predictions = iproba.argmax(axis=1)
        V = np.zeros(ensemble_proba.shape)
        idx = ensemble_proba.argmax(axis=2)
        V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
        V = V.sum(axis=0)

        IC = 0

        for j in range(n):
            if (predictions[j] == target[j]):                
                # case 1 (minority group)
                # label with majority votes on datapoint  = np.argmax(V[j, :]) 
                if(predictions[j] != np.argmax(V[j,:])):
                    IC = IC + (2*(np.max(V[j,:])) - V[j, predictions[j]])
                    
                else: # case 2 (majority group)
                    # calculate second largest nr of votes on datapoint i
                    sortedArray = np.sort(np.copy(V[j,:]))
                    IC = IC + (sortedArray[-2])
                    
            else:
                # case 3 (wrong prediction)
                IC = IC + (V[j,  int(target[j])]  -  V[j, predictions[j]] - np.max(V[j,:]) )
        ic_list.append(- 1.0 * IC)
    return ic_list


class LRPlusL1Refiner(nn.Module):
    """ Based on the original implementation: https://github.com/sbuschjaeger/leaf-refinement-experiments
    Made to work with AddTrees in Veritas.
    """
    def __init__(self):
        super(LRPlusL1Refiner, self).__init__()
        self.at = None
        self.leaf_weights = None
        self.tree_weights = None
        self.base_score = None
        self.alpha = None

    def set_params(self, at, alpha, regression=False):
        self.at = at
        torch_leafs = []
        torch_trees = []
        for t in at:
            leaf_vals = []
            for l_id in t.get_leaf_ids():
                leaf_vals.append(t.get_leaf_value(l_id, 0))
            leaf_vals = np.array(leaf_vals)
            torch_leafs.append(nn.Parameter(torch.from_numpy(leaf_vals)))
            # Although not absolutely necessary as this just initializes the weights
            # in the refiner, for RF regression models (REGR_MEAN) tree weights are 
            # 1/nb_trees in veritas.
            # The same holds for RF classifiers (CLF_MEAN), however since the base 
            # score is used here, the tree weights are set to 1.
            if at.get_type() == veritas.AddTreeType.REGR_MEAN:
                torch_trees.append(1/len(at))
            else:
                torch_trees.append(1)
        self.leaf_weights = nn.ParameterList(torch_leafs)
        self.tree_weights= nn.Parameter(torch.Tensor(torch_trees))
        self.base_score = nn.Parameter(torch.tensor(at.get_base_score(0)))
        self.alpha = alpha
        self.regression = regression

    def forward(self, x):
        offset = 0
        y = 0
        for i, p in enumerate(self.leaf_weights):
            y += torch.matmul(x[:, offset:offset + len(p)], p) * self.tree_weights[i]
            offset += len(p)
        y += self.base_score
        if not self.regression:
            y = torch.sigmoid(y)
        return y
    
    # Method that avoids making X dense, so that it fits into memory.
    def forward_sparse(self, x_sparse):
        """
        x_sparse: scipy.sparse.csr_matrix
        Returns: np.ndarray of predictions
        """
        leafs_np = [p.detach().numpy() for p in self.leaf_weights]
        tree_weights_np = self.tree_weights.detach().numpy()
        W = np.concatenate([
            leaf * tree_weight
            for leaf, tree_weight in zip(leafs_np, tree_weights_np)
        ])
        logits = x_sparse @ W + self.base_score.item()
        if not self.regression:
            return 1 / (1 + np.exp(-logits))
        else:
            return logits
    
    def refine(self, epochs, X, Y, batch_size = 128):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            mini_batches = create_mini_batches(X, Y, batch_size, True) 
            for x, y in mini_batches:
                # Take a step of gradient descent
                optimizer.zero_grad()
                loss = criterion(self.forward(x), y)
                loss.backward()
                optimizer.step()
                # Map the updated weights to the L1 space.
                with torch.no_grad():
                    step_size = optimizer.param_groups[0]['lr']
                    prox = proximal_mapping(self.tree_weights.numpy(), self.alpha, step_size)
                    self.tree_weights.copy_(torch.from_numpy(prox))

def proximal_mapping(w, alpha, step_size):
    sign = np.sign(w)
    tmp_w = np.abs(w) - (alpha * step_size)
    tmp_w = sign * np.maximum(tmp_w, 0)
    return tmp_w


def create_mini_batches(inputs, targets, batch_size, shuffle=False):
    """ Create an mini-batch like iterator for the given inputs / target / data. Copied from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    
    Parameters
    ----------
    inputs : array-like vector or matrix 
        The inputs to be iterated in mini batches
    targets : array-like vector or matrix 
        The targets to be iterated in mini batches
    batch_size : int
        The mini batch size
    shuffle : bool, default False
        If True shuffle the batches 
    """
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        start_idx += batch_size

        yield torch.from_numpy(inputs[excerpt].todense()), torch.tensor(targets.iloc[excerpt].values,dtype=torch.float64)



if __name__ == "__main__":
    cli()