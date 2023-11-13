import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def load_test_data():
    X_test = np.load('./data/X_test.npy')
    y_test = np.load('./data/y_test.npy')
    return X_test, y_test

def load_trained_model():
    model = tf.keras.models.load_model('best_model.h5')  # Загрузка обученной модели
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_results(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae


def plot_results(y_test, y_pred, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Volume USD')
    plt.title('Actual vs. Predicted Volume USD')

    if save_path:
        plt.savefig(save_path)  # Сохранение графика в файл

    plt.show()

def calculate_daily_profit(y_test, y_pred):
    # Здесь вы можете реализовать вашу стратегию торговли на дневных интервалах (в данном случае, на 5-минутных данных)
    # Например, предположим, что вы покупаете, если предсказание больше текущей цены, и продаете, если предсказание меньше.
    # Вы также можете определить размер вашей позиции и стоимость транзакции.

    position = 0  # Изначально у вас нет позиции
    initial_balance = 10000  # Изначальный баланс

    balance = initial_balance
    daily_profit = 0  # Прибыль за день
    interval_per_day = 288  # Количество интервалов в одном дне (5 минут интервал)

    for i in range(len(y_test)):
        if i % interval_per_day == 0:
            # Если начался новый день, закрываем все позиции и начинаем снова
            balance += position * y_test[i]
            position = 0

        if y_pred[i] > y_test[i]:  # Если предсказание больше текущей цены, покупаем
            position += 1
            balance -= y_test[i]  # Вычитаем стоимость покупки из баланса

    # Закрываем все позиции в конце
    balance += position * y_test[-1]

    # Рассчитываем потенциальную прибыль
    potential_profit = balance - initial_balance

    return potential_profit
def main():
    X_test, y_test = load_test_data()
    model = load_trained_model()
    y_pred = test_model(model, X_test, y_test)
    mse, mae = evaluate_results(y_test, y_pred)

    # Вывод результатов в консоль
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')

    # Запись результатов в файл
    with open('test_result.txt', 'w') as file:
        file.write(f'Mean Squared Error (MSE): {mse}\n')
        file.write(f'Mean Absolute Error (MAE): {mae}\n')

    potential_daily_profit = calculate_daily_profit(y_test, y_pred)
    print(f'Potential Daily Profit: {potential_daily_profit}')
    # Визуализация результатов
    plot_results(y_test, y_pred, save_path='plot.png')

if __name__ == "__main__":
    main()
