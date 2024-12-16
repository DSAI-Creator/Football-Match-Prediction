import pandas as pd
from src.preprocessing.utils import train_test_split
import yaml


def main():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Import training config
    train_path = config['TRAIN SETTINGS']['TRAIN_PATH']
    train_date = config['TRAIN SETTINGS']['TRAIN_DATE']
    target_col = config['TRAIN SETTINGS']['TARGET_COL']
    date_col = config['TRAIN SETTINGS']['DATE_COL']

    plot_important_feats = config['TRAIN SETTINGS']['PLOT_FEATURES_IMPORTANCE']

    # Import dataset
    df = pd.read_csv(train_path)

    # Encoding categorical data
    df['HomeTeam_Result'] = df['HomeTeam_Result'].map({'W': 3, 'D': 1, 'L': 0})

    x_train, x_test, y_train, y_test = train_test_split(df, train_date, date_col, target_col)


main()
