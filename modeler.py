# Python 3.6.0 |Anaconda 4.3.1 (64-bit)|

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV



def evaluate_model_by_cv(model, data, target, par_range, kernel, cv):

    """
    For a given model, returns a dictionary with mean test scores corresponding parameter grids passed as a dictionary(par_range).
    Arguments:
    
    par_range(dict): each value is a range of values for parameter grid building
        Example:
        
        {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
         2: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
         3: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]}
    
    model(sklearn.model): model to evaluate, example: svm
    data(df): train data
    target(df): labels
    cv = number of folds for cross validation
    """
    
    scores_dict = {}

    for pos in par_range:

        parameters = dict(C = par_range[pos], gamma = par_range[pos], kernel = [kernel])

        grid = GridSearchCV(model, parameters, cv = cv)
        grid.fit(data, target)

        scores = [test_score for test_score in grid.cv_results_["mean_test_score"]]
        scores_dict[pos] = np.array(scores).reshape(len(par_range[pos]), len(par_range[pos]))

    return scores_dict