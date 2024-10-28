import numpy as np
import pandas as pd
from datetime import datetime

def preprocess_categorical_data(df):
    df = df.copy()
    df['Time'] = df['Date'] + ' ' + df['Time']
    df['Time'] = pd.to_datetime(df['Time'])
    df['Round'] = df['Round'].map(lambda x: int(x.split(' ')[1]))
    df['Day'] = df['Day'].map({'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7})
    df['Venue'] = df['Venue'].map({'Home': 1, 'Away': 0})
    df['Result'] = df['Result'].map({'L': 0, 'W': 3, 'D': 1})
    df['GF'] = df['GF'].astype(float).astype(int)
    df['GA'] = df['GA'].astype(float).astype(int)
    df.drop(columns=['Notes', 'Date'], inplace=True)
    return df