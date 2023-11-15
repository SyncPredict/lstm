import logging
import os

import numpy as np
import pandas as pd

from components.data_preprocessing import load_and_preprocess_data


def create_dataset(X, y, time_steps=1):
    """
    Функция для преобразования временных рядов в формат, подходящий для обучения LSTM.
    X - входные данные
    y - целевая переменная
    time_steps - количество временных шагов в ряду
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def prepare_data_for_lstm(train, test, target='market_price', time_steps=30):
    """
    Подготовка данных для обучения LSTM.
    train - обучающий набор данных
    test - тестовый набор данных
    target - целевая переменная, по умолчанию 'market_price'
    time_steps - количество временных шагов, по умолчанию 30
    """
    # Выделение целевой переменной
    train_y = train[[target]]
    test_y = test[[target]]

    # Удаление целевой переменной из входных данных
    train_X = train.drop([target], axis=1)
    test_X = test.drop([target], axis=1)

    # Создание наборов данных для обучения LSTM
    train_X, train_y = create_dataset(train_X, train_y, time_steps)
    test_X, test_y = create_dataset(test_X, test_y, time_steps)

    return train_X, train_y, test_X, test_y





def process_data(data):
    train, test = load_and_preprocess_data(data)
    prepared_data = prepare_data_for_lstm(train, test)

    logging.info("Обработанные данные успешно созданы.")

    return prepared_data