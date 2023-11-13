import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import logging
import talib
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
    # Обработка пропущенных значений в 'price_usd' и 'volume_usd'
    return dataframe[dataframe['price_usd'].notna() & dataframe['volume_usd'].notna()]

def remove_outliers(dataframe: pd.DataFrame, method='iqr', z_thresh=3) -> pd.DataFrame:
    if method == 'z_score':
        z_scores = np.abs(stats.zscore(dataframe))
        filtered_entries = (z_scores < z_thresh).all(axis=1)
    elif method == 'iqr':
        Q1 = dataframe.quantile(0.25)
        Q3 = dataframe.quantile(0.75)
        IQR = Q3 - Q1
        filtered_entries = ~((dataframe < (Q1 - 1.5 * IQR)) | (dataframe > (Q3 + 1.5 * IQR))).any(axis=1)
    else:
        raise ValueError(f"Неверный метод удаления выбросов: {method}")
    return dataframe[filtered_entries]

def interpolate_missing_values(dataframe: pd.DataFrame, method='linear') -> pd.DataFrame:
    # Интерполяция только после обработки пропущенных значений
    return dataframe.interpolate(method=method)

def add_technical_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['SMA'] = talib.SMA(dataframe['price_usd'], timeperiod=25)
    dataframe['EMA'] = talib.EMA(dataframe['price_usd'], timeperiod=25)
    dataframe['RSI'] = talib.RSI(dataframe['price_usd'], timeperiod=14)
    dataframe['MACD'], dataframe['MACD_signal'], _ = talib.MACD(dataframe['price_usd'], fastperiod=12, slowperiod=26, signalperiod=9)
    return dataframe

def scale_data(dataframe: pd.DataFrame, scaler=MinMaxScaler()) -> pd.DataFrame:
    # Масштабирование данных
    scaled_data = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled_data, columns=dataframe.columns)
def preprocess_data(file_path: str) -> pd.DataFrame:
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = interpolate_missing_values(df)
    df = add_technical_indicators(df)
    df = df.dropna()
    df = scale_data(df)
    return df

if __name__ == "__main__":
    file_path = './results.json'
    processed_data = preprocess_data(file_path)
    print(processed_data.head())
