import sys
sys.path.append('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction')


import yaml


def Classifier_Model_Setting(MODEL,TARGET_COL='HomeTeam_Result', config_path='D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config.yaml'):
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

        print(f"Updated TEST SETTINGS in {config_path}, {MODEL} successfully.")

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

def main():
    Classifier_Model_Setting(MODEL='XGBClassifier')
    
if __name__ == "__main__":
    main()





