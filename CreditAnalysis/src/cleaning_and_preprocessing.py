import os
import variables
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from joblib import dump

def import_raw_data(path: str):
    return pd.read_csv(path)

def transform_binary(df: pd.DataFrame, column_list: list):
    for col in column_list:
        df[col] = (df[col] == 'yes').astype(int)

def create_target(df: pd.DataFrame, max_share: float):
    # Creates a mask for all rows who received a card and spent less than their maximum share
    mask = (df['card'] == 1) & (df['share'] <= max_share)
    df.loc[df[mask].index, 'target'] = 1
    df['target'].fillna(0, inplace=True)

def remove_columns(df: pd.DataFrame, column_list: list):
    df.drop(column_list, axis=1, inplace=True)

def treat_outliers(df: pd.DataFrame, bin_features: list):
    for col in df.columns:
        if col not in bin_features and col != 'target':
            q1 = df[col].quantile(.25)
            q3 = df[col].quantile(.75)
            iqr_fence = (q3-q1) * 1.5
            upper_fence = q3 + iqr_fence
            lower_fence = q1 - iqr_fence
            df.loc[df[col] > upper_fence, col] = upper_fence
            df.loc[df[col] < lower_fence, col] = lower_fence

def save_clean_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def divide_train_test(df: pd.DataFrame, X_columns: list, y_column: str):
    X = df[X_columns]
    y = df[y_column]
    return train_test_split(X, y, test_size=.2, random_state=42)

def normalizer(X_train: pd.DataFrame, X_test: pd.DataFrame):
    norm = PowerTransformer().fit(X_train)
    return norm.transform(X_train), norm.transform(X_test)

def scaler(X_train: pd.DataFrame, X_test: pd.DataFrame):
    minmax = MinMaxScaler().fit(X_train)
    return minmax.transform(X_train), minmax.transform(X_test)

def save_preprocessed_data(pickle_objects_list: list, pickle_names_list: list, folder_path: str):
    pickle_paths = [os.path.join(folder_path, f'{pickle}.pickle') for pickle in pickle_names_list]
    for i in range(len(pickle_objects_list)):
        dump(pickle_objects_list[i], pickle_paths[i])

def main():
    # Initial cleaning
    df = import_raw_data(variables.RAW_FILE_PATH)
    transform_binary(df, variables.BIN_FEATURES)
    create_target(df, variables.MAX_SHARE)
    remove_columns(df, variables.TO_DROP_COLUMNS)
    treat_outliers(df, variables.BIN_FEATURES)
    save_clean_df(df, variables.CLEAN_FILE_PATH)

    # Model pre-processing
    X_train, X_test, y_train, y_test = divide_train_test(df, variables.FEATURE_ORDER, variables.TARGET)
    X_train[variables.CONTINUOUS_FEATURES], X_test[variables.CONTINUOUS_FEATURES] = normalizer(
        X_train[variables.CONTINUOUS_FEATURES], X_test[variables.CONTINUOUS_FEATURES]
    )
    # Excluding binary columns
    X_train[variables.FEATURE_ORDER[:6]], X_test[variables.FEATURE_ORDER[:6]] = scaler(
        X_train[variables.FEATURE_ORDER[:6]], X_test[variables.FEATURE_ORDER[:6]]
    )
    save_preprocessed_data([X_train, X_test, y_train, y_test], variables.PICKLE_NAMES_LIST, variables.DATA_PATH)
    
if __name__ == "__main__":
    main()