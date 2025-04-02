import os
import numpy as np
import random
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from scipy.fft import fft, fftfreq
import torch.nn as nn
from flask_cors import CORS
import pywt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn   
from pytorch_wavelets import DWT1DForward, DWT1DInverse
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB
app.secret_key = 'supersecretkey'

class EnhancedWaveletDenoise(nn.Module):
    def __init__(self, wavelet='db4', levels=2):
        super().__init__()
        self.dwt = DWT1DForward(wave=wavelet, J=levels)
        self.idwt = DWT1DInverse(wave=wavelet)
        self.levels = levels
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(levels)
        ])

    def adaptive_threshold(self, coeff, level):
        threshold = self.thresholds[level]
        return torch.sign(coeff) * torch.relu(torch.abs(coeff) - threshold)

    def forward(self, x):
        # 输入形状: [B, T, 2]
        x = x.permute(0, 2, 1)  # 转换为 [B, 2, T]
        yl, yh = self.dwt(x)
        
        for l in range(self.levels):
            yh[l] = self.adaptive_threshold(yh[l], l)
        
        denoised = self.idwt((yl, yh))[:, :, :x.shape[-1]]
        return denoised.permute(0, 2, 1)  # 恢复为 [B, T, 2]

def process_signal(features):
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    denoiser = EnhancedWaveletDenoise(levels=3).to(device)
    
    processed_features = []
    for sample in features:
        # 预处理：合并I/Q分量并转换为张量
        I = sample[:1024]
        Q = sample[1024:]
        sample_tensor = torch.from_numpy(np.stack([I, Q], axis=-1)).float().unsqueeze(0).to(device)
        
        # 降噪处理
        with torch.no_grad():
            denoised_tensor = denoiser(sample_tensor)
        
        # 后处理：拆分I/Q分量并转换为NumPy
        denoised_I = denoised_tensor[0, :, 0].cpu().numpy()
        denoised_Q = denoised_tensor[0, :, 1].cpu().numpy()
        
        processed_sample = np.concatenate([denoised_I, denoised_Q])
        processed_features.append(processed_sample)
    
    return np.array(processed_features)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'npy', 'mat', 'csv', 'dat', 'wav'}

def compute_spectrum(signal, fs=1024):
    n = len(signal)
    freq = fftfreq(n, 1/fs)[:n//2]
    spectrum = np.abs(fft(signal)[:n//2]) / n
    return freq.tolist(), spectrum.tolist()

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # 确保上传目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 保存文件
            file.save(save_path)
            
            # 仅返回上传成功响应，不返回任何数据
            return jsonify({
                'message': '文件上传成功',
                'filename': filename,
                'path': save_path
            })
            
        except Exception as e:
            return jsonify({
                'error': '文件保存失败',
                'details': str(e)
            }), 500
        
        

@app.route('/api/process', methods=['POST'])
def handle_process():
    try:
        # 获取请求数据
        data = request.get_json()
        filepath = data['filepath']
        
        # 验证文件存在性
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # 加载数据
        raw_data = np.load(filepath, allow_pickle=True).item()
        features = raw_data['features']
        
        # 验证数据格式
        if features.shape[1] != 2048:
            return jsonify({'error': 'Invalid feature dimension'}), 400

        # 信号处理
        processed_features = process_signal(features)

        # 准备可视化数据（取第一个样本）
        sample_idx = 0
        
        # 原始数据
        original_I = features[sample_idx, :1024]
        original_Q = features[sample_idx, 1024:]
        
        # 处理后的数据
        processed_I = processed_features[sample_idx, :1024]
        processed_Q = processed_features[sample_idx, 1024:]

        # 生成响应数据
        response_data = {
            'original': {
                'timeDomain': {
                    'I': {'x': np.arange(1024).tolist(), 'y': original_I.tolist()},
                    'Q': {'x': np.arange(1024).tolist(), 'y': original_Q.tolist()}
                },
                'spectrum': {
                    'I': compute_spectrum(original_I),
                    'Q': compute_spectrum(original_Q)
                },
                'constellation': {
                    'I': original_I[::10].tolist(),
                    'Q': original_Q[::10].tolist()
                }
            },
            'processed': {
                'timeDomain': {
                    'I': {'x': np.arange(1024).tolist(), 'y': processed_I.tolist()},
                    'Q': {'x': np.arange(1024).tolist(), 'y': processed_Q.tolist()}
                },
                'spectrum': {
                    'I': compute_spectrum(processed_I),
                    'Q': compute_spectrum(processed_Q)
                },
                'constellation': {
                    'I': processed_I[::10].tolist(),
                    'Q': processed_Q[::10].tolist()
                }
            }
        }

        return jsonify(response_data)

    except KeyError as e:
        return jsonify({'error': f'Missing key in data: {str(e)}'}), 400
    except IOError as e:
        return jsonify({'error': f'File read error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

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