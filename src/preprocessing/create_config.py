import yaml


def create_config():
    d = {
        'TRAIN SETTINGS': {
            'TRAIN_DATE': '2023-08',
            'TRAIN_PATH': 'E:\Workspaces\My Projects\Football-Match-Prediction\data\processed\df_GD_as_target.csv',
            'DATE_COL': 'Time',
            'TARGET_COL': 'GD_Home2Away',
            'PLOT_FEATURES_IMPORTANCE': False
        },
        'TEST SETTINGS': {
            'HOME_TEAM': 'Barcelona',
            'AWAY_TEAM': 'Real Marid'
        }
    }

    with open('../../config.yaml', 'w') as file_w:
        yaml.dump(d, file_w, default_flow_style=False)

