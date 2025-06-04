import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import config
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from feature_engineering import prepare_features

def train_model(X, y, model_params):
    """Обучает модель классификации и возвращает метрики"""

    model = RandomForestClassifier(**model_params)
    model.fit(X, y)
    
    train_preds = model.predict(X)
    
    precision = precision_score(y, train_preds, average=None, zero_division=0)
    recall = recall_score(y, train_preds, average=None, zero_division=0)
    f1 = f1_score(y, train_preds, average=None, zero_division=0)
    f1_macro = f1_score(y, train_preds, average='macro', zero_division=0)
    
    return model, precision, recall, f1, f1_macro


def print_metrics(report, dataset_name):
    """Печатает метрики в удобном формате"""

    print(f"\n{dataset_name} METRICS:")
    print("=" * 60)
    print(classification_report(report['true'], report['preds'], target_names=report['classes'], digits=4))
    print("=" * 60)

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    full_df = pd.concat([train_df, test_df])
    le = LabelEncoder()
    le.fit(full_df['anomaly'])
    
    train_df['anomaly_encoded'] = le.transform(train_df['anomaly'])
    test_df['anomaly_encoded'] = le.transform(test_df['anomaly'])
    
    X_train, y_train = prepare_features(train_df, for_train=True)
    X_test, y_test = prepare_features(test_df, for_train=True)
    
    X_train['anomaly_encoded'] = train_df.loc[X_train.index, 'anomaly_encoded']
    X_test['anomaly_encoded'] = test_df.loc[X_test.index, 'anomaly_encoded']
    
    y_train = X_train.pop('anomaly_encoded')
    y_test = X_test.pop('anomaly_encoded')
    
    model, train_precision, train_recall, train_f1, train_f1_macro = train_model(
        X_train, y_train, config.RF_PARAMS
    )
    
    test_preds = model.predict(X_test)
    
    model_path = "models/random_forest_model.pkl"
    dump(model, model_path)
    encoder_path = "models/label_encoder.pkl"
    dump(le, encoder_path)
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")
    
    # Формирование отчетов
    train_report = {
        'true': y_train,
        'preds': model.predict(X_train),
        'classes': le.classes_
    }
    
    test_report = {
        'true': y_test,
        'preds': test_preds,
        'classes': le.classes_
    }
    
    print_metrics(train_report, "TRAIN")
    print_metrics(test_report, "TEST")
    
    test_f1_scores = f1_score(y_test, test_preds, average=None, zero_division=0)
    anomaly_indices = np.where(le.classes_ != 'normal')[0]
    low_performance = []
    
    for i in anomaly_indices:
        if test_f1_scores[i] < 0.7:
            low_performance.append(le.classes_[i])
    
    if low_performance:
        print("\nМинимальные метрики не достигнуты!:")
        for anomaly in low_performance:
            idx = np.where(le.classes_ == anomaly)[0][0]
            print(f"  - {anomaly}: F1={test_f1_scores[idx]:.4f}")
    else:
        print("\nМинимальные метрики достигнуты!")