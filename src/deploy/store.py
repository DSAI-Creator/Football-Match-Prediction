import sys
sys.path.append('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction')

import yaml
from utils import  Classifier_Model_Setting
#from src.tests.boosting_regressor import boosting_regressor
from src.tests.boosting_classifier import boosting_classifier

def get_classfier_model(MODEL):
    Classifier_Model_Setting(TARGET_COL='HomeTeam_Result', MODEL=MODEL, config_path='D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config_classifier.yaml')
    print('Finish')
    model, scaler, pca, cols = boosting_classifier()
    
    return model, scaler, pca, cols
'''
def get_regressor_model(MODEL):
    Regressor_Model_Setting(MODEL=MODEL)
    model, scaler, pca,cols = boosting_regressor()
    
    return model,scaler,pca,cols
'''
def save_model(name_list, type):
    print("start save model")
    if type == 'Classifier':
        model_storage = {}
        
        for model_name in name_list:
            model, scaler, pca, cols = get_classfier_model(MODEL=model_name)
            model_storage[model_name] = {
                'model': model,
                'scaler': scaler,
                'pca': pca,
                'cols': cols
            }
    '''        
    else:
        model_storage = {}
        
        for model_name in name_list:
            model, scaler, pca, cols = get_regressor_model(MODEL=model_name)
            model_storage[model_name] = {
                'model': model,
                'scaler': scaler,
                'pca': pca,
                'cols': cols
            }
    '''
    return model_storage
    
        

def main():
    # Danh sách các mô hình
    classifier_models = ['DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier']
    #regressor_models = ['DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'XGBRegressor']
    # Dictionary để lưu thông số của từng mô hình
    classifier_model_storage = save_model(classifier_models, type='Classifier')
    #regressor_model_storage = save_model(regressor_models, type='Regressor')

    # Kết hợp cả hai loại mô hình vào một dictionary
    model_storage = {
        'Classifiers': classifier_model_storage,
        #'Regressors': regressor_model_storage
    }

    # Lưu dictionary vào một file YAML
    with open('model_storage.yaml', 'w') as yaml_file:
        yaml.dump(model_storage, yaml_file, default_flow_style=False)
        
if __name__ == "__main__":
    main()

