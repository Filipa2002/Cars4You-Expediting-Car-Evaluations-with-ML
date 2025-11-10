

from typing import Callable, Dict, List
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

def evaluate_model_repeated_kfold(model, raw_train_data:pd.DataFrame,preprocessor,
                                  n_splits:int=10, n_repeats:int=10, random_state:int=42, metric:Callable=mean_squared_error):
    """
    Evaluates a model using Repeated K-Fold Cross-Validation, returning the 
    mean RMSE and its 95% confidence interval.
    
    Args:
        model: The instantiated scikit-learn model.
        raw_train_data (pd.DataFrame): The original raw training data.
        preprocessor (object): An object with a .transform(raw_data) method 
                               that returns (X, y) pandas objects.
        n_splits (int): Number of folds (K) in K-Fold.
        n_repeats (int): Number of times the K-Fold process is repeated.
        random_state (int): Seed for shuffling data.
        
    Returns:
        tuple: (mean_score, conf_interval)
    
    raises:
        AssertionError: If preprocessor does not have a .transform method.
        AssertionError: If metric is not callable.
        AssertionError: If model does not have fit and predict methods.
        
    
    """
    
    assert hasattr(preprocessor, 'transform'), "Preprocessor must have a .transform(raw_data) method."
    assert callable(metric), "Metric must be a callable function."
    assert hasattr(model, 'fit') and hasattr(model, 'predict'), "Model must have fit and predict methods."
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    total_folds = n_splits * n_repeats
    scores = np.zeros(total_folds)
    
    
    # 2. Iterate through all folds
    for fold_index, (train_index, test_index) in enumerate(tqdm.tqdm(rkf.split(raw_train_data), total=total_folds, desc="Evaluating Folds")):
        
        # Split data for the current fold
        X_rkf_train, y_rkf_train = preprocessor.fit_transform(raw_train_data.iloc[train_index])
        X_rkf_test, y_rkf_test = preprocessor.transform(raw_train_data.iloc[test_index])
        
        # Train, Predict, and Score
        model.fit(X_rkf_train, y_rkf_train)
        y_pred = model.predict(X_rkf_test)
        
        # Calculate score for the current fold using the provided metric
        score = metric(y_rkf_test, y_pred)
        scores[fold_index] = score
    
    # 3. Calculate statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Calculate 95% Confidence Interval (using 1.96 standard errors)
    conf_interval = 1.96 * (std_score / np.sqrt(len(scores)))
    
    return mean_score, conf_interval


def run_hyperparameter_tuning(train_data:pd.DataFrame,
                              preprocessor,
                              list_of_dictionaries:List[Dict],
                              metric:Callable=mean_squared_error,
                              model_class:Callable=RandomForestRegressor,
                              n_splits:int=2,
                              n_repeats:int=2,
                              random_state:int=42):
    """
    Performs Repeated K-Fold tuning for RandomForestRegressor and plots results.
    Args:
        train_data (pd.DataFrame): The original raw training data.
        preprocessor (object): An object with a .transform(raw_data) method that returns (X, y) pandas objects.
        list_of_dictionaries (list): List of hyperparameter dictionaries to evaluate.
        model_class: The scikit-learn model class to instantiate.
        n_splits (int): Number of folds (K) in K-Fold. Default is 2 for quicker testing.
        n_repeats (int): Number of times the K-Fold process is repeated. Default is 2 for quicker testing.
    Returns:
        tuple: (scores, best_params)
            - scores: List of tuples (mean_score, conf_interval) for each hyperparameter set.
            - best_params: The hyperparameter dictionary with the best mean score.
    Raises:
        AssertionError: If preprocessor does not have a .transform method.
        AssertionError: If metric is not callable.
        AssertionError: If model_class is not a valid scikit-learn estimator.
    """ 

    # Define the parameter grid to search
    
    scores = []
    print(f"--- Starting {model_class.__name__} Hyperparameter Tuning ({n_splits}x{n_repeats} Folds) ---")
    
    # Perform Grid Search
    for  _ , combination_params in enumerate(list_of_dictionaries):
            
            # Instantiate the model
            model = model_class(**combination_params)
            
            # Evaluate
            mean_score, conf_interval = evaluate_model_repeated_kfold(model, 
                                                                     train_data, 
                                                                     preprocessor,
                                                                     n_splits=n_splits, 
                                                                     n_repeats=n_repeats,
                                                                     random_state=random_state,
                                                                        metric=metric)  
            
            # Store and Print Results
            scores.append((mean_score, conf_interval))
            print(f"Result: {combination_params}-> {metric.__name__}={mean_score:.4f} ± {conf_interval:.4f}")
            
    # --- Plotting the Results ---
    print(f"\n--- Generating Plot {model_class.__name__} ---")
    labels = [f'{config_}'for config_ in list_of_dictionaries ]
    mean_scores = [v[0] for v in scores]
    conf_intervals = [v[1] for v in scores]

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x, mean_scores, yerr=conf_intervals, capsize=5, color='teal', alpha=0.8)
    
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('RMSE (Root Mean Squared Error)')
    plt.title(f'{model_class.__name__} Hyperparameter Tuning Results (Repeated K-Fold)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Determine the best parameters
    best_params = list_of_dictionaries[np.argmin([s[0] for s in scores])]
    best_rmse  = min([s[0] for s in scores])
    
    print(f"\nOptimization Complete. Best Parameters: {best_params} with Mean RMSE={best_rmse:.4f}")
    
    return scores, best_params