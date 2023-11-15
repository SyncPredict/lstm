import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_lstm_model(input_shape):
    """
    Создание модели LSTM для предсказания цены на рынке.

    Параметры:
    input_shape - форма входных данных, например (30, n_features)
    """
    model = Sequential()

    # Первый слой LSTM с Dropout
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Второй слой LSTM
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))

    # Полносвязный слой для предсказания
    model.add(Dense(25))
    model.add(Dense(1))  # Один выход для предсказания цены

    # Компиляция модели
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def get_callbacks():
    """
    Функция для получения обратных вызовов модели.
    """
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    model_checkpoint = ModelCheckpoint('best_model.tf', monitor='val_loss', mode='min',save_format="tf", save_best_only=True)

    return [early_stopping, model_checkpoint]


def create_model(data):
    train_X, train_y, test_X, test_y = data
    number_of_features = 5  # 'hash_rate', 'difficulty', 'miners_revenue', 'market_cap', 'total_bitcoins'

    input_shape = (train_X.shape[1], number_of_features)  # Например, (30, 7)
    model = create_lstm_model(input_shape)

    # Получение обратных вызовов
    callbacks = get_callbacks()

    # Обучение модели
    history = model.fit(
        train_X, train_y,
        epochs=100,
        batch_size=32,
        validation_data=(test_X, test_y),
        callbacks=callbacks,
        verbose=1
    )

    # Сохранение окончательной модели
    model.save('final_model.tf',save_format="tf")

    # При необходимости, можно также сохранить историю обучения
    np.save('./training_history.npy', history.history)
    logging.info("Модель успешно сохранена.")


