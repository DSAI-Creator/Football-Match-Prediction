import sys
sys.path.append('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction')

import streamlit as st
import pandas as pd
import time
import random
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from teamname import get_team_names
from utils import Regressor_Model_Setting, Classifier_Model_Setting
from src.tests.boosting_regressor import boosting_regressor, get_regressor_input
from src.tests.boosting_classifier import boosting_classifier, get_classifier_input

def call_classifier_model(MODEL):
    Classifier_Model_Setting(TARGET_COL='HomeTeam_Result', MODEL=MODEL, config_path='D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config_classifier.yaml')
    model, scaler, pca, cols = boosting_classifier()
    return model, scaler, pca, cols

def call_regressor_model(MODEL):
    Regressor_Model_Setting(TARGET_COL='GD_Home2Away', MODEL=MODEL, config_path='D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/config_regressor.yaml')
    model, scaler, pca, cols = boosting_regressor()
    return model, scaler, pca, cols

def map_result(result):
    if result == 2:
        return "Win"
    elif result == 1:
        return "Draw"
    elif result == 0:
        return "Lost"
    else:
        return "Unknown"  # In case of unexpected results

# Initial Model
# Classifier
gradient_boosting_classifier_model, classifier_scaler, classifier_pca, classifier_cols = call_classifier_model(MODEL='DecisionTreeClassifier')

# Regressor
xgb_regressor_model, regressor_scaler, regressor_pca, regressor_cols = call_regressor_model(MODEL='DecisionTreeRegressor')

# Tạo giao diện Streamlit
st.title("Football Match Outcome Predictor ⚽⚽⚽")

# Lấy danh sách đội từ teamname.py
try:
    team_names = get_team_names()
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()

# Nhập thông tin trận đấu
st.sidebar.header("Input Match Details")
home_team = st.sidebar.selectbox("Select Home Team", team_names, index=0)
away_team = st.sidebar.selectbox("Select Away Team", team_names, index=1)
match_date = st.sidebar.date_input("Match Date")
match_datetime = match_date.strftime("%Y %m %d")


# Nút thực hiện dự đoán
if st.sidebar.button("Predict"):
    with st.spinner("Processing..."):
        # Simulate loading for 3 seconds
        time.sleep(3)
    # Gọi các mô hình dự đoán   
    
    # Classifier
    df_team = pd.read_csv('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/src/tests/df_team.csv')
    df_opponent = pd.read_csv('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/src/tests/df_opponent.csv')
    x_test_classifier = get_classifier_input(df_team, df_opponent, home_team, away_team, match_datetime)
    x_test_classifier.drop(['AwayTeam_GA', 'AwayTeam_GF',  'AwayTeam_Result', 'HomeTeam_GA', 'HomeTeam_GF', 'HomeTeam_Result'], axis=1, inplace=True)
    x_test_classifier = x_test_classifier[classifier_cols]
    if classifier_scaler:
        x_test_classifier = classifier_scaler.transform(x_test_classifier)
    if classifier_pca:
        x_test_classifier = classifier_pca.transform(x_test_classifier)
        
    result_1 = gradient_boosting_classifier_model.predict(x_test_classifier)
    result_1 = map_result(result_1)
            
    # Regressor
    x_test_regressor = get_regressor_input(df_team, df_opponent, home_team, away_team, match_datetime)
    x_test_regressor.drop(['AwayTeam_GA', 'AwayTeam_GF',  'AwayTeam_Result', 'HomeTeam_GA', 'HomeTeam_GF', 'HomeTeam_Result'], axis=1, inplace=True)
    x_test_regressor = x_test_regressor[regressor_cols]
    if regressor_scaler:
        x_test_regressor = regressor_scaler.transform(x_test_regressor)
    if regressor_pca:
        x_test_regressor = regressor_pca.transform(x_test_regressor)
    margin = xgb_regressor_model.predict(x_test_regressor)[0]  # Predict from XGBRegresso
    
    goal_differences = [f"{margin:.1f}" for margin in [margin]]

    # Hiển thị kết quả
    st.markdown("<h2 style='text-align: center;'>Prediction Results</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 8, 2])

    # Hiển thị logo đội nhà
    with col1:
        home_logo_path = f"logo/{home_team}.png"
        if os.path.exists(home_logo_path):
            st.image(home_logo_path, use_container_width=False, width=150)
        else:
            st.write("No logo available")

    # Hiển thị bảng kết quả
    # Hai bảng ở giữa
    with col2:
        # Bảng kết quả dự đoán
        st.markdown("### Match Outcome Prediction")
        result_df = pd.DataFrame({
            "Model": ["Random Forest Classifier"],
            "Prediction": [result_1]
        })
        st.table(result_df)

        # Bảng hiệu số bàn thắng
        st.markdown("### Predicted Goal Difference (Home - Away)")
        margin_df = pd.DataFrame({
            "Model": ["XG Boost Regressor"],
            "Goal Difference": [f"{margin:.1f}"]
        })
        st.table(margin_df)

    # Hiển thị logo đội khách
    with col3:
        away_logo_path = f"logo/{away_team}.png"
        if os.path.exists(away_logo_path):
            st.image(away_logo_path, use_container_width=False, width=150)
        else:
            st.write("No logo available")
else:
    st.info("Please select match details and click 'Predict' to get the results.")

# Phần hiển thị bổ sung
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("This app uses 5 different models to predict the outcome of a football match, including the goal difference (Home - Away). Replace the placeholder prediction functions with your trained models.")
