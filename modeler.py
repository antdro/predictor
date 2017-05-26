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
        
        {
        0: {"c" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "gamma" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        }
    
    model(sklearn.model): model to evaluate, example: svm
    data(df): train data
    target(df): labels
    cv = number of folds for cross validation
    """
    
    scores_dict = {}

    for pos in par_range:
        
        c_range = par_range[pos]["c"]
        gamma_range = par_range[pos]["gamma"]

        parameters = dict(C = c_range, gamma = gamma_range, kernel = [kernel])

        grid = GridSearchCV(model, parameters, cv = cv)
        grid.fit(data, target)

        scores = [test_score for test_score in grid.cv_results_["mean_test_score"]]
        scores_dict[pos] = np.array(scores).reshape(len(c_range), len(gamma_range))

    return scores_dict



def move_scores_to_df(scores_dict, par_range):
        
    """
    Takes a dictionary with test scores and returns a dictionary with corresponding dataframes ready for drawing.
    """    
        
    scores_df = {}
    
    for pos in scores_dict:
        
        df = pd.DataFrame(scores_dict[pos])
        df.index = par_range[pos]["c"]
        df.columns = par_range[pos]["gamma"]
        df = df[::-1]
        
        scores_df[pos] = df
        
    return scores_df