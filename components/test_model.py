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

    # Расчет абсолютных ошибок и их стандартного отклонения
    abs_errors = np.abs(y_test - y_pred)
    std_dev = np.std(abs_errors)

    # Установим порог как одно стандартное отклонение
    threshold = std_dev

    # Расчет процента предсказаний в пределах порога
    correct_predictions = abs_errors <= threshold
    accuracy = np.mean(correct_predictions) * 100
    total_predictions = len(y_test)

    return mse, mae, accuracy, total_predictions, threshold

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


def main():
    X_test, y_test = load_test_data()
    model = load_trained_model()
    y_pred = test_model(model, X_test, y_test)

    logging.info(f"Shape of y_test: {y_test.shape}")
    logging.info(f"Shape of y_pred: {y_pred.shape}")

    mse, mae, accuracy, total_predictions, threshold = evaluate_results(y_test, y_pred)
    logging.info(f'Accuracy within threshold: {accuracy}%')
    logging.info(f'Threshold: {threshold}%')
    logging.info(f'Total predictions: {total_predictions}')

    logging.info(f'Mean Squared Error (MSE): {mse}')
    logging.info(f'Mean Absolute Error (MAE): {mae}')

    with open('test_result.txt', 'w') as file:
        file.write(f'Mean Squared Error (MSE): {mse}\n')
        file.write(f'Mean Absolute Error (MAE): {mae}\n')

    plot_results(y_test, y_pred, save_path='plot.png')


if __name__ == "__main__":
    main()
