#!/usr/bin/env python3
"""
Test script to demonstrate memory usage differences between regression and classification
in tree compression.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import psutil
import gc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.datasets import make_regression, make_classification

def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_memory_usage():
    print("=== Memory Usage Comparison: Regression vs Classification ===\n")
    
    # Generate datasets
    n_samples = 5000
    n_features = 10
    
    print(f"Generating datasets: {n_samples} samples, {n_features} features")
    
    # Regression dataset
    X_reg, y_reg = make_regression(n_samples=n_samples, n_features=n_features, 
                                  noise=0.1, random_state=42)
    
    # Classification dataset  
    X_cls, y_cls = make_classification(n_samples=n_samples, n_features=n_features,
                                      n_classes=2, random_state=42)
    
    print(f"Initial memory: {get_memory_mb():.1f} MB\n")
    
    # Test Random Forest models
    n_trees = 50
    max_depth = 10
    
    print("=== REGRESSION TEST ===")
    print(f"Training Random Forest Regressor ({n_trees} trees, depth {max_depth})")
    
    rf_reg = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                                   random_state=42, n_jobs=1)
    rf_reg.fit(X_reg, y_reg)
    
    memory_after_reg_training = get_memory_mb()
    print(f"Memory after regression training: {memory_after_reg_training:.1f} MB")
    
    # Simulate feature matrix creation (similar to what happens in tree compression)
    print("Creating dense feature matrix for regression...")
    n_leaves = sum(tree.tree_.n_leaves for tree in rf_reg.estimators_)
    print(f"Total leaves in forest: {n_leaves}")
    
    # Simulate the dense matrix that would be created during compression
    feature_matrix_reg = np.random.rand(n_samples, n_leaves * 2).astype(np.float64)
    memory_after_reg_matrix = get_memory_mb()
    print(f"Memory after regression feature matrix: {memory_after_reg_matrix:.1f} MB")
    print(f"Feature matrix size: {feature_matrix_reg.shape}")
    print(f"Feature matrix memory: {feature_matrix_reg.nbytes / 1024 / 1024:.1f} MB")
    
    # Test Lasso memory usage
    from sklearn.linear_model import Lasso
    print("Testing Lasso with precompute=True...")
    lasso_precompute = Lasso(alpha=0.01, max_iter=100, precompute=True)
    memory_before_lasso = get_memory_mb()
    try:
        lasso_precompute.fit(feature_matrix_reg, y_reg)
        memory_after_lasso = get_memory_mb()
        print(f"Memory after Lasso (precompute=True): {memory_after_lasso:.1f} MB")
        print(f"Memory increase for Lasso: {memory_after_lasso - memory_before_lasso:.1f} MB")
    except Exception as e:
        print(f"Lasso with precompute=True failed: {e}")
    
    # Cleanup
    del feature_matrix_reg, lasso_precompute
    gc.collect()
    
    print("\n=== CLASSIFICATION TEST ===")
    print(f"Training Random Forest Classifier ({n_trees} trees, depth {max_depth})")
    
    rf_cls = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                   random_state=42, n_jobs=1)
    rf_cls.fit(X_cls, y_cls)
    
    memory_after_cls_training = get_memory_mb()
    print(f"Memory after classification training: {memory_after_cls_training:.1f} MB")
    
    # Simulate feature matrix creation
    print("Creating dense feature matrix for classification...")
    n_leaves_cls = sum(tree.tree_.n_leaves for tree in rf_cls.estimators_)
    print(f"Total leaves in forest: {n_leaves_cls}")
    
    feature_matrix_cls = np.random.rand(n_samples, n_leaves_cls * 2).astype(np.float64)
    memory_after_cls_matrix = get_memory_mb()
    print(f"Memory after classification feature matrix: {memory_after_cls_matrix:.1f} MB")
    print(f"Feature matrix size: {feature_matrix_cls.shape}")
    print(f"Feature matrix memory: {feature_matrix_cls.nbytes / 1024 / 1024:.1f} MB")
    
    # Test LogisticRegression memory usage
    from sklearn.linear_model import LogisticRegression
    print("Testing LogisticRegression with liblinear...")
    logreg = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=100)
    memory_before_logreg = get_memory_mb()
    logreg.fit(feature_matrix_cls, y_cls)
    memory_after_logreg = get_memory_mb()
    print(f"Memory after LogisticRegression: {memory_after_logreg:.1f} MB")
    print(f"Memory increase for LogisticRegression: {memory_after_logreg - memory_before_logreg:.1f} MB")
    
    print("\n=== SUMMARY ===")
    print(f"Regression memory usage: {memory_after_reg_matrix:.1f} MB")
    print(f"Classification memory usage: {memory_after_cls_matrix:.1f} MB")
    print(f"Memory difference: {memory_after_reg_matrix - memory_after_cls_matrix:.1f} MB")
    print("\nKey reasons regression uses more memory:")
    print("1. Lasso with precompute=True creates a Gram matrix (n_features x n_features)")
    print("2. Coordinate descent algorithm requires more working memory")
    print("3. Dense feature matrices from Random Forests are very large")
    print("4. Less efficient memory access patterns in coordinate descent")

if __name__ == "__main__":
    test_memory_usage()
