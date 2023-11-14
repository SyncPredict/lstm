import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import logging
import talib
from scipy.stats.mstats import winsorize

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_json(file_path, orient='index')
        df = df.stack().apply(pd.Series)
        df.reset_index(inplace=True)
        df.rename(columns={'level_0': 'date', 'level_1': 'time'}, inplace=True)

        # Извлечение только даты из объектов Timestamp в столбце 'date'
        df['date'] = df['date'].dt.date

        # Преобразование 'date' и 'time' в строки
        df['date'] = df['date'].astype(str)
        df['time'] = df['time'].dt.time.astype(str)


        # Конкатенация 'date' и 'time' и преобразование в datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
        df.set_index('datetime', inplace=True)
        df.drop(['date', 'time'], axis=1, inplace=True)

        logging.info("Данные успешно загружены и отсортированы.")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        raise




def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Заполнение последним известным значением
    dataframe['price_usd'].fillna(method='ffill', inplace=True)
    dataframe['volume_usd'].fillna(method='ffill', inplace=True)
    return dataframe


from scipy.stats.mstats import winsorize

def limit_outliers(dataframe: pd.DataFrame, limits=(0.05, 0.05)) -> pd.DataFrame:
    # Ограничиваем выбросы для каждого столбца по отдельности
    for col in dataframe.columns:
        dataframe[col] = winsorize(dataframe[col], limits=limits)
    return dataframe


def interpolate_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.interpolate(method='time')


def add_technical_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['SMA'] = talib.SMA(dataframe['price_usd'], timeperiod=25)
    dataframe['EMA'] = talib.EMA(dataframe['price_usd'], timeperiod=25)
    dataframe['RSI'] = talib.RSI(dataframe['price_usd'], timeperiod=14)
    dataframe['MACD'], dataframe['MACD_signal'], _ = talib.MACD(dataframe['price_usd'], fastperiod=12, slowperiod=26, signalperiod=9)
    return dataframe


def scale_data(dataframe: pd.DataFrame, scaler=RobustScaler()) -> pd.DataFrame:
    scaled_data = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled_data, columns=dataframe.columns)

def preprocess_data(file_path: str) -> pd.DataFrame:
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = limit_outliers(df)
    df = interpolate_missing_values(df)
    df = add_technical_indicators(df)
    df = df.dropna()
    df = scale_data(df)
    return df

if __name__ == "__main__":
    file_path = './results.json'
    processed_data = preprocess_data(file_path)
    print(processed_data.head())
