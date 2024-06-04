import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_selection import import_train_test, combine_xs_ys, choose_optmized_or_simplest
import joblib
import variables
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.ensemble import VotingClassifier

def import_winner(winner_path: str):
    return joblib.load(winner_path)

def classification_breakdown(model, X_test, y_test, title: str):
    # predicts
    y_pred = model.predict(X_test)

    # Prints breakdown
    print(f"Classification Report for {title}:")
    print("---"*20)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Declined", "Accepted"]).plot()
    plt.title(f"Confusion Matrix for {title}:")
    plt.show()

def winners_voting(winners: dict, X_train, y_train, voting_type='hard'):
    # collecting estimators
    models = [winner for winner in winners.values()]
    estimators = []
    for i in range(3):
        estimators.append((f"clf{i+1}", models[i]))
    
    # instantiating and training voting classifier
    voting_classifier = VotingClassifier(estimators=estimators, voting=voting_type)
    voting_classifier.fit(X_train, y_train)

    return voting_classifier

def combine_test_with_clean_data(clean_data_path, X_test, y_test, columns_from_clean = ['card', 'expenditure']):
    clean_data = pd.read_csv(clean_data_path, usecols=columns_from_clean)
    test_data = pd.concat([X_test, y_test], axis=1)
    combined = pd.merge(test_data, clean_data, left_index=True, right_index=True)
    return combined

def add_predictions_to_combined(model, combined, X_test_columns):
    combined["prediction"] = model.predict(combined[X_test_columns])

def misclassification_report(combined):
    total_rows = combined.shape[0]

    zeros_mask = (combined['card'] == 0) & (combined['prediction'] != 0)
    print(f"Zeros Misclassified: {combined.loc[zeros_mask].shape[0]/total_rows * 100: .2f}%")

    ones_mask = (combined['target'] == 1) & (combined['prediction'] != 1)
    print(f"Ones Misclassified: {combined.loc[ones_mask].shape[0]/total_rows * 100: .2f}%")

    high_spenders_mask = (combined['target'] == 0) & (combined['prediction'] != 0) & (combined['expenditure'] != 0)
    print(f"High Spenders Misclassified: {combined.loc[high_spenders_mask].shape[0] / total_rows * 100: .2f}%")

    correct_classifications = combined['target'] == combined['prediction']
    print(f"Correct Classifications: {combined.loc[correct_classifications].shape[0] / total_rows * 100 : .2f}%")

def estimate_default(combined):
    # Sets as minimum default all yearly expenditure with credit card that exceeds yearly income
    combined.loc[combined['card']==1, 'estimated_default'] = (combined['income'] * 10000 - combined['expenditure'] * 12) * - 1
    combined['estimated_default'].fillna(0, inplace=True)
    combined.loc[combined['estimated_default'] < 0, 'estimated_default'] = 0
    
    # evaluating
    sns.histplot(combined['estimated_default'], kde=True)
    plt.title('Estimated Yearly Minimum Default in Test Data')
    plt.show()


def default_evaluation(combined):
    old_mask = (combined['card'] == 1) & (combined["expenditure"] * 10000 / combined['income'] > 1/12)
    new_mask = old_mask & (combined['prediction']==1)
    pass

def main():
    # initial imports
    winner = import_winner(variables.MODEL_FILE_PATH)
    X_train, X_test, y_train, y_test = import_train_test(variables.DATA_PATH, variables.PICKLE_NAMES_LIST)
    X, y = combine_xs_ys(X_train, X_test, y_train, y_test)

    # Single winner evaluation
    classification_breakdown(winner, X_test, y_test, "Single Winner")

    # Combined winners evaluation
    winners = choose_optmized_or_simplest(variables.MODELS, X, y)
    winners_hard_voting = winners_voting(winners, X_train, y_train)
    winners_soft_voting = winners_voting(winners, X_train, y_train, voting_type='soft')
    classification_breakdown(winners_hard_voting, X_test, y_test, "Winners Hard Voting")
    classification_breakdown(winners_soft_voting, X_test, y_test, "Winners Soft Voting")

    # combining test with clean for financial analysis
    combined = combine_test_with_clean_data(variables.CLEAN_FILE_PATH, X_test, y_test)
    add_predictions_to_combined(winner, combined, X_test.columns)
    misclassification_report(combined)
    estimate_default(combined)

if __name__ == "__main__":
    main()