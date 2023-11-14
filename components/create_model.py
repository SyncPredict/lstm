import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, LayerNormalization, Conv1D, Flatten, concatenate, Attention, LeakyReLU
from keras.regularizers import l1_l2, l1, l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
X_train = np.load('./data/X_train.npy')
Y_train = np.load('./data/Y_train.npy')
X_test = np.load('./data/X_test.npy')
Y_test = np.load('./data/Y_test.npy')

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


# Параметры модели
input_shape = (X_train.shape[1], X_train.shape[2])
units = 200  # Увеличенное количество нейронов
conv_filters = 64  # Количество фильтров для CNN
kernel_size = 3  # Размер ядра для CNN
dropout_rate = 0.4  # Увеличенный коэффициент Dropout
regularizer = l1_l2(l1=0.005, l2=0.005)

# Входной слой
input_layer = Input(shape=input_shape)

# Сверточный слой
conv_layer = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', padding='same')(input_layer)
conv_layer = LayerNormalization()(conv_layer)
conv_layer = Dropout(dropout_rate)(conv_layer)

# LSTM слои
x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=regularizer))(conv_layer)
x = LayerNormalization()(x)
x = Dropout(dropout_rate)(x)

# Механизм внимания
# attention = Attention()([x, x])
# x = concatenate([x, attention])

# Дополнительный LSTM слой
x = Bidirectional(LSTM(units, return_sequences=False, kernel_regularizer=regularizer))(x)
x = LayerNormalization()(x)
x = Dropout(dropout_rate)(x)

# Полносвязные слои
x = Dense(units, activation='relu')(x)
x = Dense(units, activation=LeakyReLU(alpha=0.1))(x)  # Использование LeakyReLU
output_layer = Dense(1)(x)

# Создание модели
model = Model(inputs=input_layer, outputs=output_layer)

# Компиляция модели
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('./models/best_lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

# Обучение модели
history = model.fit(X_train, Y_train, epochs=250, batch_size=16, validation_data=(X_test, Y_test), callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Сохранение модели
model.save('./models/final_lstm_model.h5')

logging.info("Усовершенствованная и усложненная модель LSTM успешно создана и обучена.")
