import json
import requests
import pandas as pd
from datetime import datetime

def get_chart_data(chart_name):
    url = f'https://api.blockchain.info/charts/{chart_name}?timespan=all&format=json'
    response = requests.get(url)
    print(response.json())
    if response.status_code == 200:
        data = response.json()['values']
        # Преобразование timestamp в формат даты и создание списка словарей
        formatted_data = [{'date': datetime.utcfromtimestamp(item['x']).strftime('%Y-%m-%d'), 'value': item['y']} for item in data]
        return formatted_data
    else:
        return []

# Получение данных
result = get_chart_data('market-cap')

# Сохранение данных в формате JSON
with open('block_results.json', 'w') as file:
    json.dump(result, file, indent=2)
