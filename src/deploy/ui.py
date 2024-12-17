import tkinter as tk
from tkinter import ttk
from datetime import datetime

# Hàm dự đoán kết quả (Giả lập cho 5 mô hình)
def predict_result():
    home_team = home_team_entry.get()
    away_team = away_team_entry.get()
    match_time = match_time_entry.get()

    if not home_team or not away_team or not match_time:
        result_label.config(text="Vui lòng nhập đầy đủ thông tin.")
        return

    # Cập nhật thông báo "Đang xử lý..."
    result_label.config(text="Đang dự đoán kết quả...")

    # Giả lập kết quả từ 5 mô hình khác nhau
    model_1_result = f"{home_team} 2 - 1 {away_team}"
    model_2_result = f"{home_team} 1 - 1 {away_team}"
    model_3_result = f"{home_team} 3 - 0 {away_team}"
    model_4_result = f"{home_team} 2 - 2 {away_team}"
    model_5_result = f"{home_team} 0 - 1 {away_team}"

    # Hiển thị kết quả dự đoán từ các mô hình
    model_1_label.config(text=f"Model 1: {model_1_result}")
    model_2_label.config(text=f"Model 2: {model_2_result}")
    model_3_label.config(text=f"Model 3: {model_3_result}")
    model_4_label.config(text=f"Model 4: {model_4_result}")
    model_5_label.config(text=f"Model 5: {model_5_result}")

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Football Match Predictor")
root.geometry("600x600")

# Tạo các label và entry widgets cho input
match_time_label = tk.Label(root, text="Thời gian trận đấu:")
match_time_label.pack(pady=5)
match_time_entry = ttk.Entry(root)
match_time_entry.pack(pady=5)
match_time_entry.insert(0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Hiển thị thời gian hiện tại

home_team_label = tk.Label(root, text="Tên đội nhà:")
home_team_label.pack(pady=5)
home_team_entry = ttk.Entry(root)
home_team_entry.pack(pady=5)

away_team_label = tk.Label(root, text="Tên đội khách:")
away_team_label.pack(pady=5)
away_team_entry = ttk.Entry(root)
away_team_entry.pack(pady=5)

# Nút dự đoán
predict_button = ttk.Button(root, text="Dự đoán kết quả", command=predict_result)
predict_button.pack(pady=20)

# Label để hiển thị kết quả
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Label cho kết quả từ 5 mô hình
model_1_label = tk.Label(root, text="Model 1:", font=("Arial", 10))
model_1_label.pack(pady=5)
model_2_label = tk.Label(root, text="Model 2:", font=("Arial", 10))
model_2_label.pack(pady=5)
model_3_label = tk.Label(root, text="Model 3:", font=("Arial", 10))
model_3_label.pack(pady=5)
model_4_label = tk.Label(root, text="Model 4:", font=("Arial", 10))
model_4_label.pack(pady=5)
model_5_label = tk.Label(root, text="Model 5:", font=("Arial", 10))
model_5_label.pack(pady=5)

# Chạy giao diện Tkinter
root.mainloop()
