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

if __name__ == "__main__":
    main()