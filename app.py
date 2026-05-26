from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import Counter
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app, origins="*")

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

HF_REPO_ID = "saidimn/ids-cnn-cicids2017"
CACHE_DIR = Path(__file__).parent / "model_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Device
device = torch.device('cpu')  # Force CPU pour économiser la mémoire

# ══════════════════════════════════════════════════════════════════
# ARCHITECTURES CNN-1D
# ══════════════════════════════════════════════════════════════════

class CNN1D_Binary(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,  64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))

class CNN1D_Attack(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))

# ══════════════════════════════════════════════════════════════════
# LAZY LOADING - Chargement différé des modèles
# ══════════════════════════════════════════════════════════════════

models_loaded = False
scaler = None
le = None
binary_model = None
attack_model = None
num_features = None
num_attack_classes = None

def download_models():
    """Télécharge les modèles depuis Hugging Face Hub"""
    files = {
        "binary": "cnn1d_binary.pth",
        "attack": "cnn1d_attacks_only.pth",
        "scaler": "scaler.pkl",
        "encoder": "label_encoder_attacks.pkl"
    }
    
    paths = {}
    print("Downloading models from Hugging Face...")
    for key, filename in files.items():
        print("  ↓ " + filename)
        paths[key] = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            cache_dir=CACHE_DIR,
            local_dir=CACHE_DIR,
            local_dir_use_symlinks=False
        )
        print("    ✓ " + paths[key])
    
    return paths

def load_models():
    """Charge les modèles en mémoire (une seule fois)"""
    global models_loaded, scaler, le, binary_model, attack_model, num_features, num_attack_classes
    
    if models_loaded:
        return
    
    print("=" * 50)
    print("Loading models into memory...")
    print("=" * 50)
    
    paths = download_models()
    
    scaler = joblib.load(paths["scaler"])
    le = joblib.load(paths["encoder"])
    
    num_features = scaler.n_features_in_
    num_attack_classes = len(le.classes_)
    
    print("Features: " + str(num_features))
    print("Classes: " + str(list(le.classes_)))
    
    # Charge les modèles un par un pour économiser la mémoire
    print("Loading binary model...")
    binary_model = CNN1D_Binary(num_features).to(device)
    binary_model.load_state_dict(torch.load(paths["binary"], map_location=device, weights_only=True))
    binary_model.eval()
    print("  ✓ Binary model loaded")
    
    print("Loading attack model...")
    attack_model = CNN1D_Attack(num_features, num_attack_classes).to(device)
    attack_model.load_state_dict(torch.load(paths["attack"], map_location=device, weights_only=True))
    attack_model.eval()
    print("  ✓ Attack model loaded")
    
    models_loaded = True
    print("All models loaded ✓\n")

# ══════════════════════════════════════════════════════════════════
# PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════════════

def preprocess(df):
    df.columns = df.columns.str.strip()

    cols_to_drop = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP',
                    'Dst Port', 'Protocol', 'Timestamp', 'Label']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    rename_dict = {
        'Tot Fwd Pkts': 'Total Fwd Packets',
        'Tot Bwd Pkts': 'Total Backward Packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
        'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
        'Fwd Pkt Len Max': 'Fwd Packet Length Max',
        'Fwd Pkt Len Min': 'Fwd Packet Length Min',
        'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
        'Fwd Pkt Len Std': 'Fwd Packet Length Std',
        'Bwd Pkt Len Max': 'Bwd Packet Length Max',
        'Fwd Header Len': 'Fwd Header Length',
        'Bwd Header Len': 'Bwd Header Length',
        'Fwd Pkts/s': 'Fwd Packets/s',
        'Bwd Pkts/s': 'Bwd Packets/s',
        'Pkt Len Min': 'Min Packet Length',
        'Pkt Len Max': 'Max Packet Length',
        'Pkt Len Mean': 'Packet Length Mean',
        'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance',
        'FIN Flag Cnt': 'FIN Flag Count',
        'SYN Flag Cnt': 'SYN Flag Count',
        'RST Flag Cnt': 'RST Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count',
        'ACK Flag Cnt': 'ACK Flag Count',
        'URG Flag Cnt': 'URG Flag Count',
        'Pkt Size Avg': 'Average Packet Size',
        'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
        'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
        'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
        'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
        'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
        'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
        'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
        'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
        'Subflow Fwd Pkts': 'Subflow Fwd Packets',
        'Subflow Bwd Pkts': 'Subflow Bwd Packets',
        'Init Fwd Win Byts': 'Init_Win_bytes_forward',
        'Init Bwd Win Byts': 'Init_Win_bytes_backward',
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward',
    }
    df = df.rename(columns=rename_dict)
    df = df.select_dtypes(include=[np.number])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    if hasattr(scaler, 'feature_names_in_'):
        for col in scaler.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[scaler.feature_names_in_]
    else:
        while df.shape[1] < 78:
            df['missing_' + str(df.shape[1])] = 0
        df = df.iloc[:, :78]

    return scaler.transform(df.values)

# ══════════════════════════════════════════════════════════════════
# ROUTES API
# ══════════════════════════════════════════════════════════════════

@app.route('/analyze', methods=['POST'])
def analyze():
    # Charge les modèles si pas encore chargés
    load_models()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
            
        total_flows = len(df)
        X_scaled = preprocess(df)
        X = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        results = []
        with torch.no_grad():
            binary_out = binary_model(X)
            binary_pred = torch.argmax(binary_out, dim=1)

            for i in range(len(X)):
                if binary_pred[i] == 0:
                    results.append('BENIGN')
                else:
                    single = X[i].unsqueeze(0)
                    attack_out = attack_model(single)
                    attack_pred = torch.argmax(attack_out, dim=1).item()
                    results.append(le.classes_[attack_pred])

        counts = Counter(results)
        total = len(results)

        labels = list(counts.keys())
        values = list(counts.values())
        percentages = [round(v/total*100, 2) for v in values]

        attacks = {k: v for k, v in counts.items() if k != 'BENIGN'}
        benign = counts.get('BENIGN', 0)

        return jsonify({
            'total_flows': total,
            'benign_count': benign,
            'attack_count': total - benign,
            'labels': labels,
            'values': values,
            'percentages': percentages,
            'attack_types': attacks,
            'results': results[:100]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'repo': HF_REPO_ID,
        'models_loaded': models_loaded
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')
