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

def transform_train_df(dataset_path, train_date, target_col, date_col, is_plot_pca_threshold, is_plot_model_selection,
                       use_pca, use_normalize):
    # Import dataset
    df = pd.read_csv(dataset_path)
    df.drop(['AwayTeam_GF', 'HomeTeam_GF', 'GD_Home2Away', 'Season', 'Round', 'HomeTeam', 'AwayTeam', 'H2H_Away_Goals','H2H_Away_Wins',
             'H2H_Draws',  'H2H_Home_Goals',  'H2H_Home_Wins',  'H2H_Total_Matches'], axis=1, inplace=True)
    df['HomeTeam_Result'] = df['HomeTeam_Result'].map({'W': 2, 'D': 1, 'L': 0})



    # Split Train & Test dataset
    x_train, x_test, y_train, y_test = train_test_split(df, train_date, date_col, target_col, is_drop_date=True)
    cols = x_train.columns
    scaler = None
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
    pca = None
    if use_pca:
        N_COMPONENTS = 60  # This threshold is chosen based on the 'Plot PCA threshold'
        pca = PCA(n_components=N_COMPONENTS)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

    # Plot Model Selection
    if is_plot_model_selection:
        plot_classifier_selection(x_train, y_train.values)

    return x_train, x_test, y_train, y_test, scaler, pca, cols

def boosting_classifier():
    # Import config plot
    with open('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config_classifier.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get status
    is_pretrain = config['GET_PRETRAIN']
    status = 'TEST SETTINGS' if is_pretrain else 'TRAIN SETTINGS'

    # Import config
    dataset_path = config[status]['DATASET_PATH']
    train_date = config[status]['TRAIN_DATE']
    target_col = config[status]['TARGET_COL']
    date_col = config[status]['DATE_COL']
    classifier = config[status]['MODEL']
    use_pca = config[status]['USE_PCA']
    use_normalize = config[status]['USE_NORMALIZE']
    plot_important_feats = config[status]['PLOT_FEATURES_IMPORTANCE']
    plot_pca_threshold = config[status]['PLOT_PCA_THRESHOLD']
    plot_model_selection = config[status]['PLOT_MODEL_SELECTION']
    n_trials = -1 if is_pretrain else config[status]['N_TRIALS']
    file.close()

    # Split train & test dataset
    x_train, x_test, y_train, y_test, scaler, pca, cols = transform_train_df(
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
    opt = BoostingClassificationOptimize(x_train, y_train, n_trials)

    # Get model
    model = opt.get_pretrained(classifier) if is_pretrain else opt.get_model(classifier)

    # Train model
    model.fit(x_train, y_train)

    # Evaluate training results
    print("---------- TRAIN SET ----------")
    y_pred = model.predict(x_train)
    evaluate_classifier(y_train, y_pred)

    # Evaluate test results
    print("---------- TEST SET ----------")
    y_pred = model.predict(x_test)
    evaluate_classifier(y_test, y_pred)

    # Plot feature importance
    if plot_important_feats:
        opt.plot_importance_feats(feature_limits=20)
    
    return model, scaler, pca, cols

def get_latest_match_data(df_team, team_name, target_time, team_col='HomeTeam'):
    """
    Retrieves data for the latest match of a given team before a target time.

    Args:
        df_team: DataFrame containing match data.
        team_name: Name of the team to search for.
        target_time: The target time to find the latest match before.
        team_col: Column name to filter matches by team (default is 'HomeTeam').

    Returns:
        A pandas Series representing the latest match data, or None if no match is found.
    """
    # Filter matches for the specified team and before the target time
    team_matches = df_team[(df_team[team_col] == team_name) & (df_team['Time'] < target_time)]
    
    if team_matches.empty:
        return None  # No matches found for the team before the target time
    
    # Ensure matches are sorted by 'Time' in descending order
    team_matches = team_matches.sort_values(by='Time', ascending=False)
    
    # Get the latest match
    latest_match = team_matches.iloc[0]  # First row after sorting
    
    return latest_match

def get_classifier_input(df_team, df_opponent, team_name, opponent_name, target_time):
    team_match = get_latest_match_data(df_team, team_name, target_time)
    opponent_match = get_latest_match_data(df_opponent, opponent_name, target_time, team_col='AwayTeam')
    team_match = team_match.to_frame().T
    opponent_match = opponent_match.to_frame().T
    team_match = team_match.drop(['Time', 'HomeTeam', 'AwayTeam', 'Round', 'Season'], axis=1)
    opponent_match = opponent_match.drop(['Time', 'HomeTeam', 'AwayTeam', 'Round', 'Season'], axis=1)
    team_match.reset_index(drop=True, inplace=True)
    opponent_match.reset_index(drop=True, inplace=True)
    input_data = pd.concat([team_match, opponent_match], axis=1)
    return input_data


'''
# Usage Example


model, scaler, pca, cols = boosting_classifier()

df_team = pd.read_csv('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/src/tests/df_team.csv')
df_opponent = pd.read_csv('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/src/tests/df_opponent.csv')
team_name = 'Barcelona'
opponent_name = 'Real Madrid'
target_time = '2024-10-24'

x_test = get_input(df_team, df_opponent, 'Barcelona', 'Real Madrid', '2021-08-22')
x_test.drop(['AwayTeam_GA', 'AwayTeam_GF',  'AwayTeam_Result', 'HomeTeam_GA', 'HomeTeam_GF', 'HomeTeam_Result'], axis=1, inplace=True)
x_test = x_test[cols]
if scaler:
	x_test = scaler.transform(x_test)
if pca:
    x_test = pca.transform(x_test)
    
y_pred = model.predict(x_test)
print(y_pred)
'''