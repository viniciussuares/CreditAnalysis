
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# File path
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_FILE_PATH = os.path.join(DATA_PATH, "CreditCard.csv")
CLEAN_FILE_PATH = os.path.join(DATA_PATH, "CreditCardClean.csv")

# Regular variables
MAX_SHARE = 1/12
TARGET = 'target'

# Column lists
CONTINUOUS_FEATURES = ['age', 'income', 'active', 'months']
NON_BIN_FEATURES = ['reports', 'dependents']
BIN_FEATURES = ['card', 'selfemp', 'owner']
TO_DROP_COLUMNS = ['rownames', 'share']
FEATURE_ORDER = ['age', 'income', 'active', 'months', 'reports', 'dependents', 'selfemp', 'owner', 'majorcards']

# Pickle Objects
PICKLE_NAMES_LIST = ['X_train', 'X_test', 'y_train', 'y_test']

# Models and Parameters
MODELS = {
    'Logistic Regression': {
        'class': LogisticRegression(),
        'params': {
            'C': [.01, .1, 1, 10, 100],
            'class_weight': ['balanced', None],
            'solver': ['lbfgs', 'liblinear'],
            'random_state': [42]
        }

    },
    'SVC': {
        'class': SVC(),
        'params': {
            'C': [.01, .1, 1, 10, 100],
            'class_weight': ['balanced', None],
            'kernel': ['linear', 'sigmoid', 'rbf'],
            'random_state': [42]
        }
    },
    'Random Forest': {
        'class': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 150, 200],
            'class_weight': ['balanced', 'balanced_subsample', None],
            'min_samples_split': [2, 5, 10, 20],
            'oob_score': [True, False],
            'random_state': [42]
        }
    }
}