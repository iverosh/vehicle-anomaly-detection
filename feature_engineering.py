import numpy as np
import config

def prepare_features(df, for_train=False):
    """Подготовка признаков для детекции аномалий"""

    df = df.copy()
    # Рассчитанный показатель количества оборотов в минуту
    df['calc_rpm'] = (df['speed'] * 1000) / (2 * np.pi * config.WHEEL_RADIUS) / 60 
    # Отношение реального и рассчитанного показателя количества оборотов в минуту
    df['rpm_ratio'] = df['wheel_rpm'] / df['calc_rpm']
    
    # Изменение расстояния
    df['distance_diff'] = df['distance'].diff().fillna(0)
    # Изменение скорости
    df['acceleration'] = df['speed'].diff().fillna(0)
    # Изменение количества оборотов в минуту
    df['wheel_rpm_diff'] = df['wheel_rpm'].diff().fillna(0)


    # скользящие статистики
    window_size = 5
    features_for_rolling = ['speed', 'wheel_rpm', 'acceleration', 'rpm_ratio']
    for col in features_for_rolling:
        # Скользящее среднее
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean().shift(1)
        # Скользящее стандартное отклонение
        df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std().shift(1)
        # Отклонение от скользящего среднего
        df[f'{col}_deviation'] = df[col] - df[f'{col}_rolling_mean']
    
    df = df.dropna().reset_index(drop=True)
    
    X = df.drop(columns=['anomaly']).select_dtypes(include=np.number)
    
    if for_train:
        y = df['anomaly']
        return X, y
    
    return X
    
