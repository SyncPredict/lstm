import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_test_data():
    X_test = np.load('./data/X_test.npy')
    y_test = np.load('./data/y_test.npy')
    return X_test, y_test

def load_trained_model():
    model = tf.keras.models.load_model('./models/best_lstm_model.h5')
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_results(y_test, y_pred):
    # Удаление лишнего измерения из y_pred
    y_pred = y_pred.squeeze()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae

def plot_results(y_test, y_pred, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price USD')
    plt.title('Actual vs. Predicted Price USD')

    if save_path:
        plt.savefig(save_path)

    plt.show()

def calculate_potential_profit(y_test, y_pred):
    position = 0
    initial_balance = 10000
    balance = initial_balance
    transaction_cost = 0.1  # Процент комиссии за транзакцию

    for i in range(len(y_test)):
        if y_pred[i] > y_test[i]:  # Покупка
            if balance >= y_test[i]:
                position += 1
                balance -= y_test[i] * (1 + transaction_cost / 100)
        elif position > 0 and y_pred[i] < y_test[i]:  # Продажа
            balance += y_test[i] * position * (1 - transaction_cost / 100)
            position = 0

    if position > 0:
        balance += y_test[-1] * position * (1 - transaction_cost / 100)

    potential_profit = balance - initial_balance
    return potential_profit

def main():
    X_test, y_test = load_test_data()
    model = load_trained_model()
    y_pred = test_model(model, X_test, y_test)

    logging.info(f"Shape of y_test: {y_test.shape}")
    logging.info(f"Shape of y_pred: {y_pred.shape}")

    mse, mae = evaluate_results(y_test, y_pred)
    # Остальной код...


    logging.info(f'Mean Squared Error (MSE): {mse}')
    logging.info(f'Mean Absolute Error (MAE): {mae}')

    with open('test_result.txt', 'w') as file:
        file.write(f'Mean Squared Error (MSE): {mse}\n')
        file.write(f'Mean Absolute Error (MAE): {mae}\n')

    potential_profit = calculate_potential_profit(y_test, y_pred)
    logging.info(f'Potential Profit: {potential_profit}')

    plot_results(y_test, y_pred, save_path='plot.png')

if __name__ == "__main__":
    main()
