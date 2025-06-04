import os
import uuid
from flask import Flask, request, send_file, render_template
import pandas as pd
from generate_dataset import generate_data, save_graphics
from anomaly_detector import detect_anomalies

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Генерация датасета с аномалиями"""
    try:
        data = request.form
        duration = int(data.get('total_time', 60))
        anomaly_freq = int(data.get('anomaly_freq', 10))
        anomaly_len = int(data.get('anomaly_duration', 5))
        
        file_id = str(uuid.uuid4())
        csv_filename = f"{file_id}.csv"
        plot_filename = f"{file_id}.png"
        csv_path = os.path.join(RESULTS_FOLDER, csv_filename)
        plot_path = os.path.join(RESULTS_FOLDER, plot_filename)
        
        data = generate_data(duration_min=duration, anomaly_freq=anomaly_freq, anomaly_len=anomaly_len)
        data.to_csv(csv_path)
        save_graphics(data, plot_path)

        base_url = request.host_url.rstrip('/')

        return render_template("generate.html", 
                               csv_url =  f"{base_url}/download/{csv_filename}",
                               plot_url = f"{base_url}/download/{plot_filename}")
    except Exception as e:
        return str(e), 500


@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400
            
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        
        # Сохранение файла
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_upload.csv')
        file.save(temp_path)
        
        anomaly_type = request.form.get('anomaly_type', 'wheel_slip')
        
        data = pd.read_csv(temp_path)
        anomalies = detect_anomalies(data, anomaly_type)
        
        os.remove(temp_path)
        
        return render_template('detect.html',
                            anomaly_type=anomaly_type,
                            anomalies=anomalies)
        
    except Exception as e:
        return str(e), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Отдача файлов для скачивания"""
    return send_file(
        os.path.join(RESULTS_FOLDER, filename),
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)