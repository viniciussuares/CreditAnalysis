# Importing libraries
import os
import variables
import numpy as np
from scipy.stats import ttest_rel, shapiro, wilcoxon
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib

def import_train_test(folder_path: str, pickle_objects_names: list):
    objects = []
    for object_name in pickle_objects_names:
        object_path = os.path.join(folder_path, f'{object_name}.pickle')
        object = joblib.load(object_path)
        objects.append(object)
    return objects

def compare_two_scores(scores_1: list, scores_2: list, significance_level = 0.05):
    """ 
    This function does the following:
    1. Creates a new array of the differences between the scores
    2. Test to see if the new array seems to be normally distributed
    3. If yes, uses t-test to verify if differences are statistically significant. Otherwise, does wilcoxon
    """
    differences = np.array(scores_2) - np.array(scores_1)
    if shapiro(differences)[1] >= significance_level: 
        return ttest_rel(scores_1, scores_2)[1] < significance_level # Scores were significantly different
    else:
        return wilcoxon(scores_1, scores_2)[1] < significance_level # Scores were significantly different

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
        condition_to_simplest = compare_two_scores(simplest_scores, optmized_scores) or simplest_scores.mean() >= optmized_scores.mean()
        if condition_to_simplest:
            winners[model] = simplest
        else:
            winners[model] = optmized
    return winners

def choose_winner(winners: dict, X, y, significance_level=0.05):
    """
    This function compares winners to choose the best model based on simplicity and score
    If there are no difference in scores, choose the easiest model to interpret (Logistic > SVC > Randon Forest )
    If they are different, choose the one that performs the best
    """
    pass


def main():
    X_train, X_test, y_train, y_test = import_train_test(variables.DATA_PATH, variables.PICKLE_NAMES_LIST)
    winners = choose_optmized_or_simplest(variables.MODELS, X_train, y_train)
    print(winners)
    # Function not finished yet

if __name__ == "__main__":
    main()