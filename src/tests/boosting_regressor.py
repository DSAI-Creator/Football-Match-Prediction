import sys
sys.path.append('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction')

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.preprocessing.utils import train_test_split, evaluate_regressor
from src.models.models import BoostingRegressionOptimize
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.encoder import Encoders
from src.preprocessing.utils import plot_regressor_selection
import numpy as np
import yaml


def transform_train_df(dataset_path, train_date, target_col, date_col, is_plot_pca_threshold, is_plot_model_selection, use_pca, use_normalize):
    # Import dataset
    df = pd.read_csv(dataset_path)
    df.drop(['AwayTeam_GF', 'HomeTeam_GF', 'HomeTeam_Result'], axis=1, inplace=True)

    # Encoding categorical data
    encoder_cols_dict = {
        'one_hot': ['HomeTeam', 'AwayTeam']
    }
    encoders = Encoders(df, encoder_cols_dict)
    df = encoders.fit_transform()

    # Split Train & Test dataset
    x_train, x_test, y_train, y_test = train_test_split(df, train_date, date_col, target_col, is_drop_date=True)

    # Normalize dataset
    if use_normalize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Plot PCA threshold
    if is_plot_pca_threshold:
        pca = PCA()
        comp = pca.fit(x_train)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.grid()
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Explained Variance')
        sns.despine()

    # Apply PCA
    if use_pca:
        N_COMPONENTS = 75  # This threshold is chosen based on the 'Plot PCA threshold'
        pca = PCA(n_components=N_COMPONENTS)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

    # Plot Model Selection
    if is_plot_model_selection:
        plot_regressor_selection(x_train, y_train.values)

    return x_train, x_test, y_train, y_test


def boosting_regressor():
    # Import config plot
    with open('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get status
    is_pretrain = config['GET_PRETRAIN']
    status = 'TEST SETTINGS' if is_pretrain else 'TRAIN SETTINGS'

    # Import config
    dataset_path = config[status]['DATASET_PATH']
    train_date = config[status]['TRAIN_DATE']
    target_col = config[status]['TARGET_COL']
    date_col = config[status]['DATE_COL']
    regressor = config[status]['MODEL']
    use_pca = config[status]['USE_PCA']
    use_normalize = config[status]['USE_NORMALIZE']
    plot_important_feats = config[status]['PLOT_FEATURES_IMPORTANCE']
    plot_pca_threshold = config[status]['PLOT_PCA_THRESHOLD']
    plot_model_selection = config[status]['PLOT_MODEL_SELECTION']
    n_trials = -1 if is_pretrain else config[status]['N_TRIALS']
    file.close()

    # Split train & test dataset
    x_train, x_test, y_train, y_test = transform_train_df(
        dataset_path=dataset_path,
        train_date=train_date,
        target_col=target_col,
        date_col=date_col,
        is_plot_pca_threshold=plot_pca_threshold,
        is_plot_model_selection=plot_model_selection,
        use_pca=use_pca,
        use_normalize=use_normalize
    )

    # Get Optimized model
    opt = BoostingRegressionOptimize(x_train, y_train, n_trials)

    # Get model
    model = opt.get_pretrained(regressor) if is_pretrain else opt.get_model(regressor)

    # Train model
    model.fit(x_train, y_train)

    # Evaluate training results
    print("---------- TRAIN SET ----------")
    y_pred = model.predict(x_train)
    evaluate_regressor(y_train, y_pred)

    # Evaluate test results
    print("---------- TEST SET ----------")
    y_pred = model.predict(x_test)
    evaluate_regressor(y_test, y_pred)

    # Plot feature importance
    if plot_important_feats:
        opt.plot_importance_feats(feature_limits=20)


boosting_regressor()
