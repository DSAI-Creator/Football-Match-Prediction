import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def train_test_split(df, train_date, date_col, target_col, is_drop_date=False):
    """
    Splitting the Train & Test set based on the latest day of getting the Train dataset
    :param is_drop_date: Boolean to drop the date column
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

    if is_drop_date:
        x_train = x_train.drop(date_col, axis=1)
        x_val = x_val.drop(date_col, axis=1)
    return x_train, x_val, y_train, y_val


def _rps_score(outcomes, predictions):
    loss = 0
    for i, p in enumerate(predictions):
        outcome = [1 if x == outcomes[i] else 0 for x in range(3)]
        tmp = probs = outs = 0
        for j, val in enumerate(predictions[i]):
            probs += val
            outs += outcome[j]
            tmp += (probs - outs) ** 2
        loss += tmp / 2
    return 1 - (loss / len(predictions))


def plot_classifier_selection(x, y):
    rps_scorer = make_scorer(_rps_score, greater_is_better=True, needs_proba=True)
    classifiers = [
        LogisticRegression(multi_class='multinomial', solver='sag', max_iter=10),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        XGBClassifier(objective='multi:softprob', num_class=3, random_state=42)
    ]

    acc_dict = {}
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    for train_index, test_index in sss.split(x, y):
        xtrain, xtest = x[train_index], x[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(xtrain, ytrain)
            predictions = clf.predict_proba(xtest)
            acc = _rps_score(ytest, predictions)
            acc_dict[name] = acc_dict.get(name, 0) + acc

    # Average the scores
    log = [{'Classifier': clf, 'Score': acc / 10.0} for clf, acc in acc_dict.items()]
    log_df = pd.DataFrame(log)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score', y='Classifier', data=log_df, color="b")
    plt.xlabel('Ranked Probability Score')
    plt.title('Model Selection Based on RPS')
    plt.show()


def plot_regressor_selection(x, y):
    """
    Evaluate and plot the performance of various regression models.

    :param x: Features dataset
    :param y: Target variable
    """
    regressors = [
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
        XGBRegressor()
    ]

    mse_dict = {}
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    for train_index, test_index in sss.split(x, y):
        xtrain, xtest = x[train_index], x[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        for reg in regressors:
            name = reg.__class__.__name__
            reg.fit(xtrain, ytrain)
            predictions = reg.predict(xtest)
            mse = mean_squared_error(ytest, predictions)
            mse_dict[name] = mse_dict.get(name, 0) + mse

    # Average the scores
    log = [{'Regressor': reg, 'MSE': mse / 10.0} for reg, mse in mse_dict.items()]
    log_df = pd.DataFrame(log)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='MSE', y='Regressor', data=log_df, color="b")
    plt.xlabel('Mean Squared Error')
    plt.title('Regressor Selection Based on MSE')
    plt.show()

def evaluate_classifier(y_val, y_pred):
    recall = recall_score(y_val, y_pred, average='macro')
    precision = precision_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Recall (Macro): {recall}")
    print(f"Precision (Macro): {precision}")
    print(f"F1 Score (Macro): {f1}")
    print(f"Accuracy: {accuracy}")


def evaluate_regressor(y_val, y_pred):
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"RMSE: {rmse}")
    print(f"R² score: {r2}")