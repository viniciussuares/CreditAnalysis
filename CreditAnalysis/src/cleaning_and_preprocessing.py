import variables
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

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
    df.to_csv(path)

def main():
    df = import_raw_data(variables.RAW_FILE_PATH)
    transform_binary(df, variables.BIN_FEATURES)
    create_target(df, variables.MAX_SHARE)
    remove_columns(df, variables.TO_DROP_COLUMNS)
    treat_outliers(df, variables.BIN_FEATURES)
    save_clean_df(df, variables.CLEAN_FILE_PATH)
    
if __name__ == "__main__":
    main()