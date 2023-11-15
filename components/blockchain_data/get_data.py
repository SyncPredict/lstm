import json

from .get_chart import fetch_correct_chart_data, fetch_floating_chart_data


def fetch_market_price():
    data = fetch_correct_chart_data('market-price')
    return data


def fetch_hash_rate():
    data = fetch_correct_chart_data('hash-rate')
    return data


def fetch_difficulty():
    data = fetch_correct_chart_data('difficulty')
    return data


def fetch_miners_revenue():
    data = fetch_correct_chart_data('miners-revenue')
    return data


def fetch_n_unique_addresses():
    data = fetch_correct_chart_data('n-unique-addresses')
    return data


def fetch_market_cap():
    data = fetch_floating_chart_data('market-cap')
    return data


def fetch_total_bitcoins():
    data = fetch_floating_chart_data('total-bitcoins')
    return data


def fetchAllData():
    # Словарь функций
    data_functions = {
        'market_price': fetch_market_price,
        'hash_rate': fetch_hash_rate,
        'difficulty': fetch_difficulty,
        'miners_revenue': fetch_miners_revenue,
        # 'unique_addresses': fetch_n_unique_addresses,
        'market_cap': fetch_market_cap,
        'total_bitcoins': fetch_total_bitcoins,
    }

    # Инициализация объединенных данных
    combined_data = {}

    # Обход всех функций и их данных
    for key, func in data_functions.items():
        dataset = func()
        for record in dataset:
            date = record['date']
            if date not in combined_data:
                combined_data[date] = {}
            combined_data[date][key] = record['value']

    return combined_data
