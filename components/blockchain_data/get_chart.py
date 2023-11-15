import json
import requests
import pandas as pd
from datetime import datetime, timedelta


def create_end_timestamp():
    now = datetime.now()
    time_24_hours_ago = now - timedelta(hours=24)
    timestamp_24_hours_ago = time_24_hours_ago.timestamp()
    return  timestamp_24_hours_ago
def fetch_correct_chart_data(chart,start_timestamp=1293840000, end_timestamp=None):
    """
    Получает и обрабатывает корректные метрики за заданный период.
    """
    if end_timestamp is None:
        end_timestamp = int(create_end_timestamp())

    all_data = []
    while start_timestamp < end_timestamp:
        url = f'https://api.blockchain.info/charts/{chart}?start={start_timestamp}&timespan=1years&format=json'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['values']
            formatted_data = [{'date': datetime.utcfromtimestamp(item['x']).strftime('%Y-%m-%d'), 'value': item['y']} for item in data]
            all_data.extend(formatted_data)
        start_timestamp += 31536000  # Переход к следующему году
    return all_data

def fetch_floating_chart_data(chart,start_timestamp=1293840000, end_timestamp=None):
    """
    Получает и обрабатывает метрики с плавающими датами за заданный период.
    """
    if end_timestamp is None:
        end_timestamp = int(create_end_timestamp())

    all_data = []
    while start_timestamp < end_timestamp:
        url = f'https://api.blockchain.info/charts/{chart}?start={start_timestamp}&timespan=1years&format=json'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['values']
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['x'], unit='s').dt.date
            grouped = df.groupby('date')['y'].mean().reset_index()
            grouped['date'] = grouped['date'].astype(str)
            formatted_data = grouped.rename(columns={'y': 'value'}).to_dict(orient='records')
            all_data.extend(formatted_data)
        start_timestamp += 31536000  # Переход к следующему году
    return all_data
