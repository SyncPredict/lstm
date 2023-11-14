import json

import requests
import pandas as pd
from datetime import datetime

charts = [
    {
        "name": "Market Price (USD)",
        "code": "market-price"
    },
    {
        "name": "Market Capitalization (USD)",
        "code": "market-cap"
    },
    {
        "name": "Exchange Trade Volume (USD)",
        "code": "trade-volume"
    },
    {
        "name": "Total Circulating Bitcoin",
        "code": "total-bitcoins"
    },
    {
        "name": "Total Hash Rate (TH/s)",
        "code": "hash-rate"
    },
    {
        "name": "Network Difficulty",
        "code": "difficulty"
    },
    {
        "name": "Miners Revenue (USD)",
        "code": "miners-revenue"
    },
    {
        "name": "Unique Addresses Used",
        "code": "n-unique-addresses"
    },
    {
        "name": "Confirmed Payments Per Day",
        "code": "n-transactions-per-block"
    },
    {
        "name": "Transaction Rate Per Second",
        "code": "transactions-per-second"
    },
    # {
    #     "name": "200 Week Moving Average Heatmap",
    #     "code": "200w-moving-avg-heatmap"
    # },
    # {
    #     "name": "Market Value to Realised Value (MVRV)",
    #     "code": "mvrv"
    # },
    # {
    #     "name": "Network Value to Transactions (NVT)",
    #     "code": "nvt"
    # },
    # {
    #     "name": "Network Value to Transactions Signal (NVTS)",
    #     "code": "nvts"
    # }
]

def get_chart_data(chart_name):
    url = f'https://api.blockchain.info/charts/{chart_name}?timespan=all&format=json'
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json()['values'])
    else:
        return pd.DataFrame()

# Список метрик

# Получить данные для каждой метрики
chart_data = {chart['code']: get_chart_data(chart['code']) for chart in charts}

# Объединить данные
combined_data = pd.DataFrame()
for chart, data in chart_data.items():
    if not data.empty:
        data['date'] = pd.to_datetime(data['x'], unit='s')


        data.set_index('date', inplace=True)
        combined_data[chart] = data['y']

# Сконвертировать данные в нужный формат
result = combined_data.to_dict(orient='index')

# Преобразовать ключи в строки даты
result = {datetime.strftime(k, '%Y-%m-%d %H:%M:%S'): v for k, v in result.items()}

# Вывод или сохранение результата
print(result)

with open('block_results.json', 'w') as file:
    json.dump(result, file, indent=2)
