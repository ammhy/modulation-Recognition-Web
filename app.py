import os
import numpy as np
import random
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB
app.secret_key = 'supersecretkey'

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # 生成模拟信号数据
        time_points = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * time_points) + np.random.normal(0, 0.2, 1000)
        
        freq_points = np.linspace(0, 50, 500)
        spectrum = np.exp(-(freq_points - 10)**2 / 5) + np.random.normal(0, 0.05, 500)
        
        return jsonify({
            'timeDomain': {
                'x': time_points.tolist(),
                'y': signal.tolist()
            },
            'spectrum': {
                'x': freq_points.tolist(),
                'y': spectrum.tolist()
            }
        })
    
    return jsonify({'error': 'File upload failed'}), 500

@app.route('/api/process', methods=['POST'])
def process_signal():
    data = request.json
    
    # 生成处理后的模拟数据
    time_points = np.linspace(0, 1, 1000)
    processed_signal = np.sin(2 * np.pi * 10 * time_points) * np.exp(-time_points/2)
    
    freq_points = np.linspace(0, 50, 500)
    processed_spectrum = np.exp(-(freq_points - 10)**2 / 3)
    
    # 生成星座图数据
    original_constellation = [[random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)] for _ in range(200)]
    processed_constellation = [
        [np.round(x)+random.uniform(-0.3, 0.3), np.round(y)+random.uniform(-0.3, 0.3)] 
        for x, y in original_constellation
    ]
    
    return jsonify({
        'processedSignal': {
            'x': time_points.tolist(),
            'y': processed_signal.tolist()
        },
        'processedSpectrum': {
            'x': freq_points.tolist(),
            'y': processed_spectrum.tolist()
        },
        'originalConstellation': original_constellation,
        'processedConstellation': processed_constellation
    })

@app.route('/api/recognize', methods=['POST'])
def recognize():
    return jsonify({
        'type': 'QPSK',
        'probabilities': {
            'BPSK': 0.05,
            'QPSK': 0.85,
            'FSK': 0.03,
            'PSK8': 0.04,
            'QAM16': 0.02,
            'QAM64': 0.01
        },
        'parameters': {
            'carrierFrequency': "1000.00",
            'symbolRate': "250.00",
            'rolloffFactor': "0.35",
            'modulationIndex': "1.00"
        },
        'reliability': 4.2
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)