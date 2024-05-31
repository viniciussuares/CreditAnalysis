# Importing libraries
import os
import variables
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, shapiro, wilcoxon, levene, f_oneway, kruskal
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib

def import_train_test(folder_path: str, pickle_objects_names: list):
    objects = []
    for object_name in pickle_objects_names:
        object_path = os.path.join(folder_path, f'{object_name}.pickle')
        object = joblib.load(object_path)
        objects.append(object)
    return objects

def combine_xs_ys(X_train, X_test, y_train, y_test):
    return pd.concat([X_train, X_test]), pd.concat([y_train, y_test])

def perform_randomized_search(model_class, model_params, X, y):
    optmized_model = RandomizedSearchCV(
        model_class, 
        model_params, 
        n_iter=30, 
        cv=5, 
        n_jobs=-1, 
        scoring='f1_weighted')
    optmized_model.fit(X, y)
    return optmized_model.best_estimator_

def compare_two_scores(scores_1: list, scores_2: list, significance_level = 0.05):
    """ 
    This function does the following:
    1. Creates a new array of the differences between the scores
    2. Test to see if the new array seems to be normally distributed
    3. If yes, uses paired t-test to verify if differences are statistically significant. Otherwise, does wilcoxon
    """
    if np.array_equal(scores_1, scores_2):
        return # optmized was simplest so the means will be the same and there will be no difference
    else:
        differences = np.array(scores_2) - np.array(scores_1)
        if shapiro(differences).pvalue >= significance_level: 
            return ttest_rel(scores_1, scores_2).pvalue < significance_level # Scores were significantly different
        else:
            return wilcoxon(scores_1, scores_2).pvalue < significance_level # Scores were significantly different

def choose_optmized_or_simplest(models: dict, X, y):
    """
    This function does the following:
    1. For each model instantiates its class and generates an optmized version
    2. Then compares how they perform against each other using the function compare_two_scores
    3. If they are no different, returns the simplest model. Otherwise, returns the one with highest score
    """
    winners = {}
    for model, class_param in models.items():
        simplest = class_param['class'] # simplest / default values
        optmized = perform_randomized_search(class_param['class'], class_param['params'], X, y)
        simplest_scores = cross_val_score(simplest, X, y, scoring="f1_weighted", n_jobs=-1, cv=10)
        optmized_scores = cross_val_score(optmized, X, y, scoring="f1_weighted", n_jobs=-1, cv=10)
        condition_to_simplest = (
            not compare_two_scores(simplest_scores, optmized_scores) # no significant difference between models
            or simplest_scores.mean() >= optmized_scores.mean() # or simplest performs better or equally
        )
        if condition_to_simplest:
            winners[model] = simplest
        else:
            winners[model] = optmized
    return winners

def compare_two_winners(models: list, scores: list):
    """
    This function compare two winners after verifying there is a difference between the 3 models
    If there's no difference between 2 winners, choose the simplest (Logistic Regression > SVC > Random Forest)
    """
    mean_scores = [np.mean(score) for score in scores]
    max_index = np.argmax(mean_scores)
    middle_index = [index for index in range(3) if index not in [max_index, np.argmin(mean_scores)]][0]
    # choose two with greatest means
    if compare_two_scores(scores[max_index], scores[middle_index]): # max and middle are different
        return models[max_index]
    else:
        if "Logistic Regression" in [models[max_index], models[middle_index]]:
            return "Logistic Regression"
        else:
            return "SVC"

def choose_winner(winners: dict, X, y, significance_level=0.05):
    """
    This function compares winners to choose the best model based on simplicity and score
    If there are no difference in scores, choose the easiest model to interpret (Logistic > SVC > Randon Forest )
    If they are different, choose the one that performs the best
    """
    # cross val for model
    models = [winner for winner in winners.keys()]
    scores = [cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring='f1_weighted') for model in winners.values()]

    # checking normality
    normality = np.array([shapiro(score).pvalue for score in scores])

    # checking homogeneity
    homogeneity = levene(*scores).pvalue
    
    if min(normality.min(), homogeneity) >= significance_level:
        # performs anova as assumptions were true
        if f_oneway(*scores).pvalue >= significance_level:
            return winners["Logistic Regression"] # No difference between models, choose the simplest
        else:
            # Check if the difference is between the 2 with greatest mean scores
            return winners[compare_two_winners(models, scores)]
    else:
        # performs kruskal as assumptions were false
        if kruskal(*scores).pvalue >= significance_level:
            return winners["Logistic Regression"] # No difference between models, choose the simplest
        else:
            # Check if the difference is between the 2 with greatest mean scores
            return winners[compare_two_winners(models, scores)]

def train_grand_winner(grand_winner, X, y):
    return grand_winner.fit(X, y)

def save_grand_winner(grand_winner, path):
    joblib.dump(grand_winner, path)

def main():
    X_train, X_test, y_train, y_test = import_train_test(variables.DATA_PATH, variables.PICKLE_NAMES_LIST)
    X, y = combine_xs_ys(X_train, X_test, y_train, y_test)
    print(X, y)
    winners = choose_optmized_or_simplest(variables.MODELS, X, y)
    grand_winner = choose_winner(winners, X, y)
    print(f"The selected model was ... {grand_winner}!")
    train_grand_winner(grand_winner, X_train, y_train)
    save_grand_winner(grand_winner, variables.MODEL_FILE_PATH)

if __name__ == "__main__":
    main()