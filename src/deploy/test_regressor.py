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
from src.tests.boosting_regressor import boosting_regressor
from utils import transform_train_df_classifier, transform_train_df_regressor, Classifier_Model_Setting,Regressor_Model_Setting


def get_1_sample_regressor():
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
    x_train, x_test, y_train, y_test = transform_train_df_regressor(
        dataset_path=dataset_path,
        train_date=train_date,
        target_col=target_col,
        date_col=date_col,
        is_plot_pca_threshold=plot_pca_threshold,
        is_plot_model_selection=plot_model_selection,
        use_pca=use_pca,
        use_normalize=use_normalize
    )
    return x_train
    
def main():
    Regressor_Model_Setting(TARGET_COL='GD_Home2Away', MODEL='DecisionTreeRegressor',config_path='D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config.yaml')
    x_train_regressor = get_1_sample_regressor()
    regressor_model = boosting_regressor()
    #x_train = extract_input(date,home,away)
    game_gd = regressor_model.predict(x_train_regressor)
    print(game_gd)
    
if __name__ == '__main__':
    main()