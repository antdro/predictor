# Python 3.6.0 |Anaconda 4.3.1 (64-bit)|

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
from math import isnan
import matplotlib.pyplot as plt
import seaborn as sb



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
                
                
                
def get_best_params(csv, test_score):
    
    """
    Returns a list of pairs of parameters ensuring the best model's performance. 
    Performance is measured with mean test score.
    
    Arguments:
        csv(str): path to csv containing grid of scores
        test_score(float): value is used to filter parameters ensuring the best performance
        
    Returns: 
        params(list): list of tuples i.e. [(gamma1, c1), (gamma2, c2)]
    """
    
    c_column_name = "Unnamed: 0"

    df = pd.read_csv(csv)
    df.index = list(df[c_column_name])
    df = df.iloc[:, 1:]
    df = df[df > test_score]
    params_dict = dict(df)

    params = []

    for gamma in params_dict:
        for c in params_dict[gamma].index:
            if not isnan(params_dict[gamma][c]):
                params.append((float(gamma), float(c)))
                
    return params



def train_and_predict(data_learn, target_learn, data_predict, c, gamma):
    
    """
    Train SVC and make predictions.
    
    Arguments:
        data_learn(df): test dataset
        target_learn(series): labels for test
        data_predict(df): dataset to label
        c - c parametr for SVC
        gamma - gamma parameter for SVC
        
    Returns:
        predictions(numpy.ndarray): probabilites for each record belonning to either class
    """
    
    clf = svm.SVC(random_state = 3, kernel = 'rbf', C = c, gamma = gamma, probability = True)
    clf.fit(data_learn, target_learn)
    predictions = clf.predict_proba(data_predict)
    
    return predictions



def trade(target_predict, predictions, prob, price, stake):
    
    """
    Trade on predictions and print accuracy and profit.
    
    Arguments:
        prob(float): choose a fixture to bet on if probability for target is higher that prob value
        price(float): an average price the bet is placed at
        stake(int): the amount per bet
        target(dataframe): dataset with target fixtures results
        predictions(numpy): array with probabilities for fixtures
        
    """
    
    results = pd.DataFrame(target_predict)
    
    target_probs = [prob[1] for prob in predictions]
    
    results["predictions"] = target_probs
    
    bets = results[results.predictions > prob]
    correct_bets = bets[bets.target == 1]
    n_correct = correct_bets.shape[0]
    n_bets = bets.shape[0]

    payout = n_correct * price * stake
    bank = n_bets * stake
    
    if n_bets > 0:
        profit_proc = round(100 * (payout - bank) / bank , 2)
        accuracy = round(100 * (n_correct/n_bets), 2)

        print (str(n_correct) + " out of " + str(n_bets))
        print ("accuracy: " + str(accuracy) + "%")
        print ("profit: " + str(profit_proc) + "%")
    else:
        print("no fixtures suggested")
        
    return bets.index



def get_all_predictions_by_best_params(params, data_learn, target_learn, data_predict, target_predict, prob):
    
    """
    Returns list of indeces for fixtures predicted with best parameters.

    data_learn(df): test dataset
    target_learn(series): labels for test
    data_predict(df): dataset to label
    target_predict(series): actual results
    prob(float): choose a fixture to bet on if probability for target is higher that prob value
    """
    
    all_predictions = []

    for pair in params:

        c = float(pair[1])
        gamma = float(pair[0])
        
        predictions = train_and_predict(data_learn, target_learn, data_predict, c, gamma)
        
        results = pd.DataFrame(target_predict)  
        target_probs = [prob[1] for prob in predictions]
        results["predictions"] = target_probs
        bets = results[results.predictions > prob]
        indeces = list(bets.index)

        all_predictions = all_predictions + indeces
        
    return all_predictions



def trade_and_print_report(dfs, scores, probs, path, months):

    """
    Train classifier using the best parameters found in csvs located in path.
    Make predictions for each month starting from February.
    Trade on predictions.
    Print report for each month and the whole period of trading.

    Arguments:
        dfs(dict) - dictionary with monthly fixtures
        scores(list): each score greater than one in the list provides the set of best parameters"
        probs(list): predictions with probability higher than one from list is selected for trading
        path(str): location of csvs with parameter grids for each month
        months(list): periods of time (months) to trade on

    Return: None, but prints the report.
    
    TODO: replace printing report with saving tradin results into a file.

    """
    for prob in probs:
        bets = pd.DataFrame()
        for score in scores:

            print ("\n")
            print ("test_score: " + str(score))
            print ("prob: " + str(prob))
            print ("\n")

            learn_df = pd.concat([dfs[8], dfs[9], dfs[10], dfs[11], dfs[12], dfs[1]])

            for month in months:
                predict_df = dfs[month]

                data_learn = learn_df.iloc[:, :-1]
                target_learn = learn_df.iloc[:, -1]

                data_predict = predict_df.iloc[:, :-1]
                target_predict = predict_df.iloc[:, -1]

                params = get_best_params(path + str(month - 1) + ".csv", score)

                indeces = get_all_predictions_by_best_params(params, data_learn, target_learn, data_predict, target_predict, prob)
                indeces = list(set(indeces))

                learn_df = pd.concat([learn_df, predict_df])

                bets_df = data.loc[indeces, :]

                bets = pd.concat([bets, bets_df])

                n_bets = bets_df.shape[0]
                n_draws = bets_df[bets_df.HG == bets_df.AG].shape[0]
                n_evens = n_bets - sum((bets_df.HG + bets_df.AG) % 2)

                draws = bets_df[bets_df.HG == bets_df.AG]
                prices_draws = draws[prices]

                if n_bets != 0:
                    profit_procent = round(100* (prices_draws.sum() - n_bets) / n_bets, 1)
                else:
                    profit_procent = 0

                print ("month: " + str(month))
                print (str(n_draws) + " draws out of " + str(n_bets))
                print (str(n_evens) + " even goals out of " + str(n_bets))
                print ("profit: " + str(profit_procent) + "%")
                print ("\n")

        # overall accuracy and profit
        all_draws = bets[bets.HG == bets.AG]
        n_all_draws = all_draws.shape[0]
        n_all_bets = bets.shape[0]

        if n_all_bets == 0:
            print ("overall accuracy: 0%")
            print ("overall profit: 0%")
            print ("\n")
        else:
            print ("overall accuracy: " + str(round(100 * n_all_draws/n_all_bets, 2)) + "%")
            print ("overall profit: " + str(round(100 * (all_draws[prices].sum() - n_all_bets) / n_all_bets, 2)) + "%")
            print ("\n")