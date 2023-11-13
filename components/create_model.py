import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def load_data():
    X_train = np.load('./data/X_train.npy')
    y_train = np.load('./data/y_train.npy')
    X_test = np.load('./data/X_test.npy')
    y_test = np.load('./data/y_test.npy')
    return X_train, y_train, X_test, y_test

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
    return history

X_train, y_train, X_test, y_test = load_data()
model = build_model(X_train.shape[1:])
history = train_model(model, X_train, y_train, X_test, y_test)
