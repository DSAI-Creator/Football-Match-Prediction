import sys
sys.path.append('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction')

import yaml
from utils import Regressor_Model_Setting
from src.tests.boosting_regressor import boosting_regressor



def get_regressor_model(MODEL):
    print("Start get_regressor_model")  # Kiểm tra bước này chạy
    Regressor_Model_Setting(TARGET_COL='GD_Home2Away', MODEL=MODEL, config_path='D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config.yaml')
    print("Regressor_Model_Setting completed")  # Nếu lỗi xảy ra trước đây, chương trình sẽ không in dòng này
    model, scaler, pca, cols = boosting_regressor()
    print("boosting_regressor completed")  # Nếu lỗi xảy ra ở boosting_regressor, chương trình sẽ không in dòng này
    print('finish')
    return model, scaler, pca, cols

def save_model(name_list, type):
    model_storage = {}
    for model_name in name_list:
        print(f"Calling get_regressor_model for: {model_name}")
        model, scaler, pca, cols = get_regressor_model(MODEL=model_name)
        model_storage[model_name] = {
            'model': model,
            'scaler': scaler,
            'pca': pca,
            'cols': cols
        }
    return model_storage

        

def main():
    # Danh sách các mô hình
    regressor_models = ['DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'XGBRegressor']
    # Dictionary để lưu thông số của từng mô hình
    regressor_model_storage = save_model(regressor_models, type='Regressor')

    print(regressor_model_storage)
        
if __name__ == "__main__":
    main()

