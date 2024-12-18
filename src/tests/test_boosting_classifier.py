import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.preprocessing.utils import train_test_split
from src.models.models import BoostingClassificationOptimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.encoder import Encoders
from src.preprocessing.utils import plot_classifier_selection
import numpy as np
import yaml


def transform_train_df():
    with open('../../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Import training config
    dataset_path = config['TRAIN SETTINGS']['DATASET_PATH']
    train_date = config['TRAIN SETTINGS']['TRAIN_DATE']
    target_col = config['TRAIN SETTINGS']['TARGET_COL']
    date_col = config['TRAIN SETTINGS']['DATE_COL']
    is_plot_pca_threshold = config['TRAIN SETTINGS']['PLOT_PCA_THRESHOLD']
    is_plot_model_selection = config['TRAIN SETTINGS']['PLOT_MODEL_SELECTION']
    use_pca = config['TRAIN SETTINGS']['USE_PCA']

    # Import dataset
    df = pd.read_csv(dataset_path)
    df.drop(['AwayTeam_GF', 'HomeTeam_GF', 'GD_Home2Away'], axis=1, inplace=True)
    df['HomeTeam_Result'] = df['HomeTeam_Result'].map({'W': 2, 'D': 1, 'L': 0})

    # Encoding categorical data
    encoder_cols_dict = {
        'one_hot': ['HomeTeam', 'AwayTeam']
    }
    encoders = Encoders(df, encoder_cols_dict)
    df = encoders.fit_transform()

    # Split Train & Test dataset
    x_train, x_test, y_train, y_test = train_test_split(df, train_date, date_col, target_col, is_drop_date=True)

    # Normalize dataset
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Plot PCA threshold
    if is_plot_pca_threshold:
        pca = PCA()
        pca.fit(x_train)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.grid()
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Explained Variance')
        sns.despine()

    # Apply PCA
    if use_pca:
        N_COMPONENTS = 75 # This threshold is chosen based on the 'Plot PCA threshold'
        pca = PCA(n_components=N_COMPONENTS)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

    # Plot Model Selection
    if is_plot_model_selection:
        plot_classifier_selection(x_train, y_train.values)

    return x_train, x_test, y_train, y_test


def test_boosting_classifier():
    # Import config plot
    with open('../../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    plot_important_feats = config['TRAIN SETTINGS']['PLOT_FEATURES_IMPORTANCE']
    dataset_path = config['TRAIN SETTINGS']['DATASET_PATH']
    n_trials = config['TRAIN SETTINGS']['N_TRIALS']
    classifier = config['TRAIN SETTINGS']['MODEL']

    # Get team names
    df = pd.read_csv(dataset_path)
    team_names = df['HomeTeam'].unique()
    ones_mask = np.array([1 for _ in range(len(team_names))])

    # Split train & test dataset
    x_train, x_test, y_train, y_test = transform_train_df()

    # Get Optimized model
    opt = BoostingClassificationOptimize(x_train, y_train, n_trials)
    model = opt.get_model(classifier)

    # Train model
    model.fit(x_train, y_train)

    # Evaluate training results
    y_pred = model.predict(x_train)
    recall = recall_score(y_train, y_pred, average='macro')
    precision = precision_score(y_train, y_pred, average='macro')
    f1 = f1_score(y_train, y_pred, average='macro')
    accuracy = accuracy_score(y_train, y_pred)
    print("---------- TRAIN SET ----------")
    print(f"Recall (Macro): {recall}")
    print(f"Precision (Macro): {precision}")
    print(f"F1 Score (Macro): {f1}")
    print(f"Accuracy: {accuracy}")

    # Evaluate test results
    y_pred = model.predict(x_test)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    print("---------- TEST SET ----------")
    print(f"Recall (Macro): {recall}")
    print(f"Precision (Macro): {precision}")
    print(f"F1 Score (Macro): {f1}")
    print(f"Accuracy: {accuracy}")

    # Plot feature importance
    if plot_important_feats:
        opt.plot_importance_feats(feature_limits=20)


test_boosting_classifier()
