import streamlit as st
import pandas as pd
import random
import os

# Placeholder: Hàm dự đoán từ các mô hình (thay thế bằng mô hình thực của bạn)
def model_1_predict(home_team, away_team, match_date):
    return random.choice(["Home Win", "Away Win", "Draw"]), random.randint(-3, 3)

def model_2_predict(home_team, away_team, match_date):
    return random.choice(["Home Win", "Away Win", "Draw"]), random.randint(-3, 3)

def model_3_predict(home_team, away_team, match_date):
    return random.choice(["Home Win", "Away Win", "Draw"]), random.randint(-3, 3)

def model_4_predict(home_team, away_team, match_date):
    return random.choice(["Home Win", "Away Win", "Draw"]), random.randint(-3, 3)

def model_5_predict(home_team, away_team, match_date):
    return random.choice(["Home Win", "Away Win", "Draw"]), random.randint(-3, 3)

# Tạo giao diện Streamlit
st.title("Football Match Outcome Predictor ⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽⚽")

# Nhập thông tin trận đấu
st.sidebar.header("Input Match Details")
home_team = st.sidebar.text_input("Home Team", "Team A")
away_team = st.sidebar.text_input("Away Team", "Team B")
match_date = st.sidebar.date_input("Match Date")

# Nút thực hiện dự đoán
if st.sidebar.button("Predict"):
    # Gọi các mô hình dự đoán
    result_1, margin_1 = model_1_predict(home_team, away_team, match_date)
    result_2, margin_2 = model_2_predict(home_team, away_team, match_date)
    result_3, margin_3 = model_3_predict(home_team, away_team, match_date)
    result_4, margin_4 = model_4_predict(home_team, away_team, match_date)
    result_5, margin_5 = model_5_predict(home_team, away_team, match_date)

    # Hiển thị kết quả
    st.markdown("<h2 style='text-align: center;'>Prediction Results</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 6, 2])

    # Hiển thị logo đội nhà
    with col1:
        home_logo_path = f"logo/{home_team}.png"
        if os.path.exists(home_logo_path):
            st.image(home_logo_path, use_container_width=False, width=150)
        else:
            st.write("No logo available")

    # Hiển thị bảng kết quả
    with col2:
        results_df = pd.DataFrame({
            "Model": ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"],
            "Prediction": [result_1, result_2, result_3, result_4, result_5],
            "Goal Difference (Home - Away)": [margin_1, margin_2, margin_3, margin_4, margin_5]
        })
        st.table(results_df)

    # Hiển thị logo đội khách
    with col3:
        away_logo_path = f"logo/{away_team}.png"
        if os.path.exists(away_logo_path):
            st.image(away_logo_path, use_container_width=False, width=150)
        else:
            st.write("No logo available")

    # Điều chỉnh khoảng cách logo và bảng kết quả
    st.markdown("<style>.css-1lcbmhc {margin-top: 500px !important;}</style>", unsafe_allow_html=True)

else:
    st.info("Please enter match details and click 'Predict' to get the results.")

# Phần hiển thị bổ sung
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("This app uses 5 different models to predict the outcome of a football match, including the goal difference (Home - Away). Replace the placeholder prediction functions with your trained models.")
