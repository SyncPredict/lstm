import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
X_train = np.load('./data/X_train.npy')
Y_train = np.load('./data/Y_train.npy')
X_test = np.load('./data/X_test.npy')
Y_test = np.load('./data/Y_test.npy')

# Параметры модели
input_shape = (X_train.shape[1], X_train.shape[2])
units = 100  # Увеличение количества нейронов
dropout_rate = 0.3  # Увеличенный коэффициент исключения

# Создание модели LSTM
model = Sequential()
model.add(Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))
model.add(Bidirectional(LSTM(units, return_sequences=False, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))))  # Изменено на return_sequences=False
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))
model.add(Dense(units, activation='relu'))
model.add(Dense(1))


# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
model_checkpoint = ModelCheckpoint('./models/best_lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Обучение модели
history = model.fit(X_train, Y_train, epochs=150, batch_size=64, validation_data=(X_test, Y_test), callbacks=[early_stopping, model_checkpoint])

# Сохранение модели
model.save('./models/final_lstm_model.h5')

logging.info("Усовершенствованная модель LSTM успешно создана и обучена.")
