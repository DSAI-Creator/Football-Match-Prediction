import sys
sys.path.append('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction')

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.preprocessing.utils import train_test_split, evaluate_classifier
from src.models.models import BoostingClassificationOptimize
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.encoder import Encoders
from src.preprocessing.utils import plot_classifier_selection
import numpy as np
import yaml
import yaml
import pandas as pd
import random
import os
from teamname import get_team_names
from src.tests.boosting_classifier import boosting_classifier
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

def transform_train_df_classifier(dataset_path, train_date, target_col, date_col, is_plot_pca_threshold, is_plot_model_selection,
                       use_pca, use_normalize):
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
    if use_normalize:
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
        N_COMPONENTS = 75  # This threshold is chosen based on the 'Plot PCA threshold'
        pca = PCA(n_components=N_COMPONENTS)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

    # Plot Model Selection
    if is_plot_model_selection:
        plot_classifier_selection(x_train, y_train.values)

    # Select one sample from the test set
    if len(x_test) > 0:
        x_test = x_test[:1]  # Select the first sample
        y_test = y_test.iloc[:1]  # Select the corresponding label
        
    if len(x_train) > 0:
        x_train = x_train[:1]  # Select the first sample
        y_train = y_train.iloc[:1]  # Select the corresponding label

    return x_train, x_test, y_train, y_test

def Classifier_Model_Setting(TARGET_COL='HomeTeam_Result', MODEL='DecisionTreeClassifier', config_path='D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config.yaml'):
    """
    Updates the TARGET_COL and MODEL fields in the TEST SETTINGS section of a config.yaml file.

    Parameters:
        TARGET_COL (str): The target column to use for classification.
        MODEL (str): The classifier model to use.
        config_path (str): Path to the config.yaml file.

    Returns:
        None
    """
    try:
        # Load the config file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Update the TEST SETTINGS
        test_settings = config.get('TEST SETTINGS', {})
        test_settings['TARGET_COL'] = TARGET_COL
        test_settings['MODEL'] = MODEL
        config['TEST SETTINGS'] = test_settings

        # Save the updated config back to the file
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, sort_keys=False)

        print(f"Updated TEST SETTINGS in {config_path} successfully.")

    except FileNotFoundError:
        print(f"Error: The file {config_path} was not found.")
    except yaml.YAMLError as e:
        print(f"Error processing the YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def Regressor_Model_Setting(TARGET_COL='GD_Home2Away', MODEL='DecisionTreeRegressor', config_path='D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config.yaml'):
    """
    Updates the TARGET_COL and MODEL fields in the TEST SETTINGS section of a config.yaml file for regression.

    Parameters:
        TARGET_COL (str): The target column to use for regression.
        MODEL (str): The regressor model to use.
        config_path (str): Path to the config.yaml file.

    Returns:
        None
    """
    try:
        # Load the config file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Update the TEST SETTINGS
        test_settings = config.get('TEST SETTINGS', {})
        test_settings['TARGET_COL'] = TARGET_COL
        test_settings['MODEL'] = MODEL
        config['TEST SETTINGS'] = test_settings

        # Save the updated config back to the file
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, sort_keys=False)

        print(f"Updated TEST SETTINGS in {config_path} successfully.")

    except FileNotFoundError:
        print(f"Error: The file {config_path} was not found.")
    except yaml.YAMLError as e:
        print(f"Error processing the YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def transform_train_df_regressor(dataset_path, train_date, target_col, date_col, is_plot_pca_threshold, is_plot_model_selection, use_pca, use_normalize):
    """
    Prepares the dataset for training a regression model, selecting a single sample for testing.

    Parameters:
        dataset_path (str): Path to the dataset file.
        train_date (str): Date for splitting the train and test sets.
        target_col (str): Column name for the target variable.
        date_col (str): Column name for the date variable.
        is_plot_pca_threshold (bool): Whether to plot PCA threshold.
        is_plot_model_selection (bool): Whether to plot model selection.
        use_pca (bool): Whether to apply PCA to the dataset.
        use_normalize (bool): Whether to normalize the dataset.

    Returns:
        x_train_sample (ndarray): Single training sample features.
        x_test_sample (ndarray): Single test sample features.
        y_train_sample (scalar): Single training sample target.
        y_test_sample (scalar): Single test sample target.
    """
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

    # Select a single sample for train and test
    x_train_sample = x_train.iloc[[0]]
    x_test_sample = x_test.iloc[[0]]
    y_train_sample = y_train.iloc[0]
    y_test_sample = y_test.iloc[0]

    # Normalize dataset
    if use_normalize:
        scaler = StandardScaler()
        x_train_sample = scaler.fit_transform(x_train_sample)
        x_test_sample = scaler.transform(x_test_sample)

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
        N_COMPONENTS = 75  # This threshold is chosen based on the 'Plot PCA threshold'
        pca = PCA(n_components=N_COMPONENTS)
        pca.fit(x_train)
        x_train_sample = pca.transform(x_train_sample)
        x_test_sample = pca.transform(x_test_sample)

    # Plot Model Selection
    if is_plot_model_selection:
        plot_regressor_selection(x_train, y_train.values)

    return x_train_sample, x_test_sample, y_train_sample, y_test_sample









