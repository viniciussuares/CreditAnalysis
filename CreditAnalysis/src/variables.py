
import os

# File path
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_FILE_PATH = os.path.join(DATA_PATH, "CreditCard.csv")
CLEAN_FILE_PATH = os.path.join(DATA_PATH, "CreditCardClean.csv")

# Regular variables
MAX_SHARE = 1/12

# Column lists
CONTINUOUS_FEATURES = ['age', 'income', 'active', 'months']
NON_BIN_FEATURES = ['reports', 'dependents']
BIN_FEATURES = ['card', 'selfemp', 'owner']
TO_DROP_COLUMNS = ['rownames', 'share']
FEATURE_ORDER = ['age', 'income', 'active', 'months', 'reports', 'dependents', 'selfemp', 'owner', 'majorcards']
