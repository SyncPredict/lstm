import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Добавление временных признаков
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Интерполяция для обработки пропущенных значений
    df.interpolate(method='linear', inplace=True)

    # Проверка на стационарность и дифференцирование
    if not check_stationarity(df['volume_usd']):
        df['volume_usd_diff'] = df['volume_usd'].diff().fillna(0)

    return df

def check_stationarity(series, alpha=0.05):
    result = adfuller(series)
    return result[1] <= alpha

def create_sequences(df, target_column, window_size=5):
    sequence_data = []
    target_data = []

    for i in range(len(df) - window_size):
        sequence_data.append(df.iloc[i:(i + window_size)].values)
        target_data.append(df.iloc[i + window_size][target_column])

    return np.array(sequence_data), np.array(target_data)

def process_and_save(file_path, target_column, window_size=5, save_dir='./data'):
    df = load_and_preprocess_data(file_path)

    # Масштабирование данных с использованием StandardScaler
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    X, y = create_sequences(pd.DataFrame(scaled_df, columns=df.columns), target_column, window_size)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

if __name__ == "__main__":
    process_and_save('results.json', target_column='volume_usd')
