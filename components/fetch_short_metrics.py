import json
import datetime
import os
import time
import logging
import pandas as pd
import san
from san import AsyncBatch

# Конфигурация и ключи API
api_key = "uid4mufihquwnqoq_gbefl7bstxlshbi3"
coin_slug = 'ethereum'
initial_date_to = '2023-01-07T00:00:00Z'
results_file = 'results.json'
interval = "5m"
san.ApiConfig.api_key = api_key

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Загрузка метрик
with open('metrics.json', 'r') as file:
    metrics = json.load(file)


def decrement_date(date_str):
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    decremented_date = date_obj - datetime.timedelta(days=1)
    return decremented_date.strftime("%Y-%m-%dT%H:%M:%SZ")


def save_results(res):
    # Проверяем, существует ли файл
    if os.path.exists(results_file):
        # Открываем файл для чтения
        with open(results_file, 'r') as file:
            try:
                # Загружаем существующие данные
                existing_data = json.load(file)
            except json.JSONDecodeError:
                # Если файл пуст или содержит неверный формат, создаем пустой словарь
                existing_data = {}
    else:
        existing_data = {}

    # Обновляем данные, добавляя новые значения из res
    for date, data in res.items():
        if date in existing_data:
            for date_t, metrics_data in data.items():
                if date_t in existing_data:
                    # Обновляем существующие данные
                    existing_data[date][date_t].update(metrics_data)
                else:
                    # Добавляем новые данные
                    existing_data[date][date_t] = metrics_data
        else:
            existing_data[date] = {}
            for date_t, metrics_data in data.items():
                existing_data[date][date_t] = metrics_data

    # Перезаписываем файл с обновленными данными
    with open(results_file, 'w') as file:
        json.dump(existing_data, file, indent=2)


def fetch_data(date_to, date_from, metrics):
    try:
        batch = AsyncBatch()
        for metric in metrics:
            batch.get(
                metric,
                slug=coin_slug,
                from_date=date_from,
                to_date=date_to,
                interval=interval
            )
        result = batch.execute()
        process_result(metrics, result)

    except Exception as e:
        if san.is_rate_limit_exception(e):
            logging.error(f"Rate limit reached :{e}")
            return  # Завершение работы программы
        else:
            logging.error(f"Error requesting: {e}")
            print(e)
            return  # Завершение работы программы


def main():
    try:
        date_to = initial_date_to
        date_from = decrement_date(date_to)

        while date_from != '2019-01-07T00:00:00Z':
            fetch_data(date_to, date_from, metrics)
            date_to = date_from
            date_from = decrement_date(date_from)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return


def process_result(metrics, results):
    # Initialize the result dictionary
    final_result = {}

    # Iterate over each metric and its corresponding DataFrame
    for metric, df in zip(metrics, results):
        # Iterate over each row in the DataFrame
        for datetime, row in df.iterrows():
            # Convert datetime to string format
            date_str = datetime.strftime("%Y-%m-%d")
            datetime_str = datetime.strftime("%H:%M:%S")

            # If the datetime is not already in the final_result, add it
            if date_str not in final_result:
                final_result[date_str] = {}
            if datetime_str not in final_result[date_str]:
                final_result[date_str][datetime_str] = {}

            # Add the metric result for this datetime
            final_result[date_str][datetime_str][metric] = row['value']

    save_results(final_result)


if __name__ == "__main__":
    main()
