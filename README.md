# Football-Match-Prediction
---
## Github Commit Flow:
#### 1, Clone Repo:
```bash
git clone https://github.com/DSAI-Creator/Football-Match-Prediction.git
```

#### 2, (Local) Pull $${\color{lightblue}Main \space Branch}$$ to your $${\color{lightblue}Local \space Main \space Branch}$$ before do anything!:
```bash
git checkout main
git pull origin main
```

#### 3, (Local) Create/Switch to your $${\color{lightblue}Local}$$ branch named ***features/{your_name}***:
- Create (Skip this if you have this branch):
```bash
git checkout -b features/{your_name}
```

- Switch:
```bash
git checkout features/{your_name}
```

#### 4, (Local) Add, Commit your changes:
- Add:
($${\color{lightblue}Note}$$: '.' represent all files)
```bash
git add .
```

- Commit:
($${\color{lightblue}Note}$$: The ***{message}*** is recommended to be in the syntax "{Action} {Object}", eg. "Create dataset folder", "Hotfix {function} in data_processing.py")
```bash
git commit -m {message}
```

#### 5, (Local) Push your changes:
- Create Remote Branch:
($${\color{lightblue}Note}$$: Skip this if this REPOSITORY have your ***features/{your_name}*** branch)
```bash
git push origin --set-upstream features/{your_name}
```

- Push:
```bash
git push origin features/{your_name}
```

#### 6, (Remote) Check the Pull Request and Fix the errors (if exists)
---
## Folder Structure:
#### 1, Base Structure:
Football-Match-Prediction/
```bash
├── .idea/
├── .venv/
├── data/
│   ├── processed/
│   ├── raw/
│       └── scraping/
├── src/
│   ├── eda/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   └── scraping/
├── tests/
└── README.md
```

#### 2, Details:
- 'data' Folder (.csv files):
  + 'data/raw': contains 'all_teams_data.csv' (final scraping version), and scraping csv files in './scraping' folder.
  + 'data/processed': contains outputs of 'src/preprocessing/..' files.

- 'src' Folder (.py or .ipynb files):
  + 'src/eda': contains all eda files
  + 'src/evaluation': contains all custom evaluation metric files
  + 'src/models': contains all AI models
  + 'src/preprocessing': contains all preprocessing files (Output dir: 'data/processed')
  + 'src/scraping': contains all scraping files (Output dir: 'data/raw/scraping')

- 'tests/' Folder (.py or .ipynb files):
  + Contains all test files for each function/file in 'src/..', eg. 'tests/test_evaluation_fbmetric' or 'tests/test_ensemble_learning'
