# Python 3.6.0 |Anaconda 4.3.1 (64-bit)|

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os



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



def calculate_scores(dfs, params_grid):
    
    """
    Go though data in dfs period by period, updating training set when each period is completed.
    Calculate test score for each parameter pair in a grid specified in params_grid dictionary.
    Save results as .csv file.
    For each grid plot a graph for each month and save it as .png file.
    
    Arguments:
        
        dfs(dict): dictionary where keys - ordinary numbers of months, values - df for corresponding period
        params_grid(dict): dictionary definging grids of parameters, keys - grid name, values - dict with c, gamma ranges 
    """
    
    prev = dfs[8]

    for month in dfs.keys():

        month_str = str(month)

        if month != 8:

            learn_df = pd.concat([prev, dfs[month]])

            data_learn = learn_df.iloc[:, :-1]
            target_learn = learn_df.iloc[:, -1]

            model = svm.SVC(random_state = 3)
            scores_dict = evaluate_model_by_cv(model, data_learn, target_learn, params_grid, "rbf", 5)
            scores_df = move_scores_to_df(scores_dict, params_grid)

            for grid in scores_df:

                if not os.path.exists(grid):
                    os.makedirs(grid)

                scores_df[grid].to_csv(grid + "/" + month_str + ".csv")
                scores = scores_df[grid]

                fig, ax = plt.subplots(1, 1, figsize = (10, 8))
                g1 = sb.heatmap(scores, cbar = True, ax = ax, cmap = "hot")

                tly = ax.get_yticklabels()
                ax.set_yticklabels(tly, rotation = 0)

                tlx = ax.get_xticklabels()
                ax.set_xticklabels(tlx, rotation = 90)

                plt.title("month: " + month_str, fontsize = 16)
                plt.xlabel("gamma", fontsize = 16)
                plt.ylabel("c", fontsize = 16)

                fig.savefig(grid + "/" + month_str + ".png")
                plt.close();

                prev = learn_df