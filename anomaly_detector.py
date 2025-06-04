import pandas as pd
import numpy as np
import argparse
import config
import joblib
from feature_engineering import prepare_features



def detect_anomaly_intervals(predictions, target_label, timestamps=None, merge_gap=0):
    """
    Находит интервалы с заданной аномалией в последовательности предсказаний.
    
    Параметры:
    predictions -- массив предсказанных меток
    target_label -- числовая метка искомой аномалии
    timestamps -- массив временных меток
    merge_gap -- объединять интервалы с разрывом <= N точек (0 по умолчанию)
    
    Возвращает список интервалов в виде (start, end)
    """

    anomaly_mask = (predictions == target_label).astype(int)
    
    starts = np.where(np.diff(anomaly_mask, prepend=0) == 1)[0]
    ends = np.where(np.diff(anomaly_mask, append=0) == -1)[0]
    intervals = list(zip(starts, ends))
    
    # Объединение близких интервалов
    if merge_gap > 0 and intervals:
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] - last[1] <= merge_gap + 1:
                merged[-1] = (last[0], current[1])
            else:
                merged.append(current)
        intervals = merged
    
    return [{"start":str(timestamps[s]), "end": str(timestamps[e])} for s, e in intervals]
    


def detect_anomalies(df, anomaly_type, model_path="models"):
    model = joblib.load(f"{model_path}/random_forest_model.pkl")
    label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
    
    X = prepare_features(df)
    preds = model.predict(X)
    
    timestamps = pd.to_datetime(df['timestamp']).values
    anomaly_type_label = label_encoder.transform([anomaly_type])
    
    intervals = detect_anomaly_intervals(preds, anomaly_type_label, timestamps, merge_gap=10)

    return intervals
        






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--anomaly", type=str, required=True, choices=["wheel_slip", "gps_loss"], help="Anomaly type to detect")
    parser.add_argument("--model_path", type=str, default="models", help="Path to models directory")
    args = parser.parse_args()
    
    df = pd.read_csv(args.dataset)
    intervals = detect_anomalies(df, args.anomaly, args.model_path)
    
    print(f"\nНайдено {len(intervals)} интервалов аномалии {args.anomaly} :")
    for i, interval in enumerate(intervals, 1):
        print(f"""{i}. {interval["start"]} to {interval["end"]} """)