from data_preprocessing import preprocess_data
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

file_path = './results.json'

# Функция для создания временных рядов
def create_time_series(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), :]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Функция для разделения данных на обучающую и тестовую выборку
def train_test_split_data(data, test_size=0.2, look_back=1):
    # Нормализация данных
    data_normalized = data.values.astype('float32')
    # Разделение данных
    train_size = int(len(data_normalized) * (1 - test_size))
    train, test = data_normalized[0:train_size, :], data_normalized[train_size - look_back:, :]
    # Создание временных рядов
    X_train, Y_train = create_time_series(train, look_back)
    X_test, Y_test = create_time_series(test, look_back)
    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    prepared_data = preprocess_data(file_path)

    # Параметры для временных рядов
    look_back = 5  # Количество временных шагов для входных данных

    # Разделение данных на обучающую и тестовую выборки
    X_train, Y_train, X_test, Y_test = train_test_split_data(prepared_data, test_size=0.2, look_back=look_back)

    logging.info(f'Размер обучающей выборки: {X_train.shape}, Размер тестовой выборки: {X_test.shape}')

    # Сохранение обработанных данных
    np.save('./data/X_train.npy', X_train)
    np.save('./data/Y_train.npy', Y_train)
    np.save('./data/X_test.npy', X_test)
    np.save('./data/Y_test.npy', Y_test)

    logging.info("Обработанные данные успешно сохранены.")
