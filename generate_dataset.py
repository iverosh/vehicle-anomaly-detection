import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import config
import matplotlib.pyplot as plt

def generate_data(duration_min=60, anomaly_freq=200, anomaly_len=60):
    """
    Генерирует телеметрию автомобиля
    • wheel_rpm — число оборотов колеса в минуту.
    • speed — скорость движения (км/ч).
    • distance — пройденное расстояние по GPS (в метрах).
    С шагом 1 секунда 

    В сгенерированной телеметрии есть аномалии:
    Проскальзывание колеса (wheel_slip) - внезапное увеличение количества оборотов в минуту
    Потеря сигнала GPS (gps_loss) - пройденное расстояние перестает увеличиваться

    Какая именно будет аномалия выбирается случайно в каждый аномальный промежуток
    

    Параметры:
    duration_min -- продолжительность сгенерированных данных в минутах (не менее 60 минут)
    anomaly_freq -- частота аномалий (каждые anomaly_freq секунд)
    anomaly_len -- длина промежутков, на которых есть аномалии 
    """

    WHEEL_RADIUS = config.WHEEL_RADIUS
    MAX_SPEED = config.MAX_SPEED
    MIN_SPEED = config.MIN_SPEED
    if duration_min < 60:
        duration_min = 60

    start_time = datetime.now()
    interval = timedelta(seconds=1)
    speed = np.random.uniform(30, 150)
    wheel_rpm = (speed * 1000) / (2 * np.pi * WHEEL_RADIUS) / 60
    real_distance = 0.0
    
    data = [{
        "timestamp": start_time,
        "wheel_rpm": wheel_rpm,
        "speed": speed,
        "distance": real_distance,
        "anomaly": "normal"
    }]
    
    anomaly = "normal"
    acceleration = 0
    cur_normal_duration = anomaly_freq
    cur_anomaly_duration = 0
    
    for i in range(1, duration_min * 60):
        timestamp = start_time + i * interval
        
        if data[-1]["speed"] >= MAX_SPEED:
            acceleration = np.random.uniform(-0.3, 0)
        elif data[-1]["speed"] <= MIN_SPEED:
            acceleration = np.random.uniform(0, 0.3)
        else:
            acceleration = np.random.uniform(-0.3, 0.3)
            
        speed = max(MIN_SPEED, min(MAX_SPEED, data[-1]["speed"] + acceleration))
        wheel_rpm = (speed * 1000) / (2 * np.pi * WHEEL_RADIUS) / 60
        real_distance += speed / 3.6  # км/ч -> м/с
        cur_anomaly = "normal"

        # Генерация аномалий
        if cur_anomaly_duration > 0:
            if anomaly == "wheel_slip":
                wheel_rpm *= np.random.uniform(1.5, 2.0)
                cur_anomaly = "wheel_slip"
            elif anomaly == "gps_loss":
                real_distance = data[-1]["distance"]  # Замораживаем расстояние
                cur_anomaly = "gps_loss"
            cur_anomaly_duration -= 1
            if cur_anomaly_duration == 0:
                cur_normal_duration = anomaly_freq
        
        # Начало новой аномалии
        elif cur_normal_duration == 0:
            anomaly = np.random.choice(["wheel_slip", "gps_loss"])
            cur_anomaly = anomaly
            cur_anomaly_duration = anomaly_len
        else:
            cur_normal_duration -= 1

        data.append({
            "timestamp": timestamp,
            "wheel_rpm": wheel_rpm,
            "speed": speed,
            "distance": real_distance,
            "anomaly": cur_anomaly
        })
    
    return pd.DataFrame(data)


def save_graphics(dataset, name):
    """
    Строит графики телеметрии по pandas dataframe и сохраняет в виде изображения

    Параметры:
    dataset -- pandas dataframe сгенерированный функцией generate_data
    name -- имя изображения
    """

    count = {"normal": 0, "gps_loss": 0, "wheel_slip": 0}
    groups = {"normal": [], "gps_loss": [], "wheel_slip": []}
    colors = {"normal": "white", "gps_loss": "yellow", "wheel_slip": "red"}
    prev_anomaly = "normal"
    start = dataset["timestamp"].iloc[0]

    for i in range(1, len(dataset)):
        anomaly = dataset["anomaly"].iloc[i]
        if prev_anomaly != anomaly:
            groups[prev_anomaly].append((start, dataset["timestamp"].iloc[i-1]))
            prev_anomaly = anomaly
            start = dataset["timestamp"].iloc[i]

    plt.figure(figsize=(14, 8))
    

    count = {"normal": 0, "gps_loss": 0, "wheel_slip": 0}
    plt.subplot(311)
    plt.ylabel("Количество оборотов в минуту")
    plt.plot(dataset["timestamp"], dataset["wheel_rpm"])

    for i in groups.keys():
        for j in groups[i]:
            if count[i] == 0:
                plt.axvspan(j[0], j[1], facecolor=colors[i], alpha=0.3, label=i)  
                count[i] += 1
            else:
                plt.axvspan(j[0], j[1], facecolor=colors[i], alpha=0.3)
    plt.grid()
    plt.legend(loc=2)


    count = {"normal": 0, "gps_loss": 0, "wheel_slip": 0}
    plt.subplot(312)
    plt.ylabel("Скорость (км/ч)")
    plt.plot(dataset["timestamp"], dataset["speed"])
    for i in groups.keys():
        for j in groups[i]:
            if count[i] == 0:
                plt.axvspan(j[0], j[1], facecolor=colors[i], alpha=0.3, label=i)  
                count[i] += 1
            else:
                plt.axvspan(j[0], j[1], facecolor=colors[i], alpha=0.3)

    plt.grid()
    plt.legend(loc=2)


    count = {"normal": 0, "gps_loss": 0, "wheel_slip": 0}
    plt.subplot(313)
    plt.ylabel("Пройденное расстояние по GPS (м)")
    plt.plot(dataset["timestamp"], dataset["distance"])
    for i in groups.keys():
        for j in groups[i]:
            if count[i] == 0:
                plt.axvspan(j[0], j[1], facecolor=colors[i], alpha=0.3, label=i)  
                count[i] += 1
            else:
                plt.axvspan(j[0], j[1], facecolor=colors[i], alpha=0.3)
    plt.grid()


    plt.legend(loc=2)
    plt.savefig(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--anomaly_freq", type=float, default=200, help="Anomaly frequency")
    parser.add_argument("--anomaly_len", type=float, default=60, help="Anomaly lenght")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Dataset filename")
    parser.add_argument("--graphics_name", type=str, default="graphics", help="Graphics filename")
    
    args = parser.parse_args()
    
    df = generate_data(args.duration, args.anomaly_freq, args.anomaly_len)
    df.to_csv(args.dataset_name, index=False)
    save_graphics(df, args.graphics_name)

    print(f"Dataset generated: {args.dataset_name}, {args.graphics_name} ({len(df)} records)")