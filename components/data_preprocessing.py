import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

def load_and_preprocess_data(raw_data):
    # Инициализация пустых DataFrame для обучающего и тестового наборов
    train = pd.DataFrame()
    test = pd.DataFrame()

    try:
        df = pd.DataFrame(raw_data)
        data = df.transpose()
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'date'}, inplace=True)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        # Проверка и обработка пропущенных значений
        if data.isnull().values.any():
            data.interpolate(method='linear', inplace=True)

        # Анализ и устранение аномалий
        z_scores = zscore(data)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        data = data[filtered_entries]

        # Нормализация данных
        scaler = RobustScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

        # Извлечение признаков
        # data_scaled['SMA_7'] = data_scaled['market_price'].rolling(window=7).mean()
        # data_scaled['SMA_30'] = data_scaled['market_price'].rolling(window=30).mean()

        # Разделение данных на обучающие и тестовые наборы
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(data_scaled):
            train, test = data_scaled.iloc[train_index], data_scaled.iloc[test_index]

    except Exception as e:
        print(f"Ошибка при загрузке или преобразовании данных: {e}")

    return train, test

