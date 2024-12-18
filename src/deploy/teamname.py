import os

def get_team_names():
    # Lấy danh sách tên đội từ folder logo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_folder = os.path.join(current_dir, 'logo')
    if not os.path.exists(logo_folder):
        raise FileNotFoundError(f"Folder '{logo_folder}' does not exist. Please check the path.")
    team_names = [os.path.splitext(file)[0] for file in os.listdir(logo_folder) if file.endswith('.png')]
    return sorted(team_names)
