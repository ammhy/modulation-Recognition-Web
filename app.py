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

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        
        if filename.endswith('.npy'):
            try:
                # 加载并验证数据
                data = np.load(save_path, allow_pickle=True).item()
                
                # 关键检查点1: 验证数据结构
                if 'features' not in data or 'labels' not in data:
                    return jsonify({'error': 'NPY文件必须包含features和labels字段'}), 400
                
                features = data['features']
                
                # 关键检查点2: 验证维度
                if features.ndim != 2 or features.shape[1] != 2048:
                    return jsonify({
                        'error': f'数据维度错误，应为(样本数, 2048)，实际为{features.shape}'
                    }), 400
                
                # 关键检查点3: 至少包含一个样本
                if features.shape[0] < 1:
                    return jsonify({'error': '数据中未包含有效样本'}), 400
                
                # 取第一个样本
                sample_index = 0
                I = features[sample_index, :1024].tolist()
                Q = features[sample_index, 1024:].tolist()
                
                # 生成信号参数
                fs = 1024  # 采样率
                t = np.arange(1024) / fs  # 时间向量
                
                # 频谱计算函数
                def compute_spectrum(signal):
                    n = len(signal)
                    freq = np.fft.fftfreq(n, 1/fs)[:n//2]
                    spectrum = np.abs(np.fft.fft(signal)[:n//2]) / n
                    return freq.tolist(), spectrum.tolist()
                
                # 计算频谱
                freq_I, spectrum_I = compute_spectrum(I)
                freq_Q, spectrum_Q = compute_spectrum(Q)
                
                # 构建响应
                return jsonify({
                    'timeDomain': {
                        'I': {'x': t.tolist(), 'y': I},
                        'Q': {'x': t.tolist(), 'y': Q}
                    },
                    'spectrum': {
                        'I': {'x': freq_I, 'y': spectrum_I},
                        'Q': {'x': freq_Q, 'y': spectrum_Q}
                    },
                    'constellation': {
                        'I': I[::10],  # 降采样
                        'Q': Q[::10]
                    }
                })
                return jsonify(response) 
            except Exception as e:
                app.logger.error(f"NPY文件处理失败: {str(e)}")
                return jsonify({'error': f'文件处理失败: {str(e)}'}), 400
            
        # 其他文件类型处理...
        elif filename.endswith('.dat'):
            # [之前的.dat处理代码]
            return jsonify(response)
        
        else:
            # 添加对其他文件类型的处理或返回错误
            return jsonify({
                'error': f'文件类型 {filename.split(".")[-1]} 暂未支持处理',
                'supported_types': ['npy', 'dat']
            }), 400
            
    except Exception as e:
        app.logger.error(f"处理异常: {str(e)}")
        return jsonify({'error': '服务器处理失败'}), 500
    finally:
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception as e:
                app.logger.error(f"删除临时文件失败: {str(e)}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mat', 'csv', 'dat', 'wav'}

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