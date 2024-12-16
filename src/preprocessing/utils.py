import pandas as pd


def train_test_split(df, train_date, date_col, target_col):
    """
    Splitting the Train & Test set based on the latest day of getting the Train dataset
    :param df: Df of dataset
    :param train_date: DateTime of the latest day of getting the Train dataset
    :param date_col: Str name of the date column
    :param target_col: Str name of the target column
    :return: (x_train, y_train), (x_test, y_test)
    """
    # Ensure df[date_col] in DateTime type
    df[date_col] = pd.to_datetime(df[date_col])

    # Split the data into training and validation sets
    train = df[df[date_col] <= train_date]
    val = df[df[date_col] > train_date]

    # Define (x,y) of train & valid dataset
    x_train = train.drop(target_col, axis=1)
    x_val = val.drop(target_col, axis=1)
    y_train = train[target_col]
    y_val = val[target_col]

    return x_train, x_val, y_train, y_val
