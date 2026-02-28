import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuration (UPDATED) ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Features used for the model (based on your previous selection)
NUM_FEATURES = ['duration', 'src_bytes', 'dst_bytes', 'num_failed_logins', 'count', 'srv_count', 'same_srv_rate']
CAT_FEATURES = ['protocol_type']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
MODEL_PATH = "models/nids_multiclass_model.joblib" # Changed model name
DATASET_PATH = "data/network_intrusions (1).csv" 

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Statistics tracking (Global for API status)
stats = {
    'total_scans': 0,
    'threats_detected': 0,
    'normal_traffic': 0,
    'accuracy': 0.0 # Will be updated on load/train
}

# --- NIDS Predictor Class ---
class NIDSPredictor:
    """Manages model loading, training, and prediction for the NIDS."""
    def __init__(self):
        self.pipeline = None
        self.is_trained = False
        self.model_accuracy = 0.0
        self.label_encoder = None
        self.target_encoder = None # NEW: Encoder for attack types
        
    def load_custom_dataset(self, file_path):
        """Load the custom dataset and prepare target for multi-class classification."""
        logger.info(f"Loading dataset from {file_path}...")
        
        try:
            # Read the comma-separated file (CSV)
            df = pd.read_csv(file_path, sep=',')
            
            # --- Data Validation and Preprocessing ---
            missing_features = [f for f in ALL_FEATURES if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features in dataset: {missing_features}")
            
            if 'attack_type' not in df.columns or df['attack_type'].nunique() < 2:
                 raise ValueError("Target column 'attack_type' is missing or lacks variation.")

            logger.info(f"✓ Dataset loaded. Records: {len(df)}. Unique Attack Types: {df['attack_type'].nunique()}.")
            return df
            
        except FileNotFoundError:
            logger.error(f"✗ Dataset file not found: {file_path}")
            raise FileNotFoundError(f"Please ensure '{os.path.basename(file_path)}' is in the 'data/' directory.")
        except Exception as e:
            logger.error(f"✗ Error loading dataset: {e}")
            raise
    
    def create_pipeline(self):
        """Create ML pipeline with preprocessing and model."""
        pipeline = Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), NUM_FEATURES),
                    # 'encoded_protocol_type' is passed through as it will be manually encoded 
                    # before passing to the pipeline during fit/predict
                    ('cat', 'passthrough', ['encoded_protocol_type'])
                ],
                remainder='drop'
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced',
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ])
        
        return pipeline
    
    def train_model(self, save_model=True):
        """Train the ML model using custom dataset for multi-class classification."""
        logger.info("="*60)
        logger.info("Starting multi-class model training...")
        
        df = self.load_custom_dataset(DATASET_PATH)
        
        # Prepare features and target (using raw attack_type)
        X = df[ALL_FEATURES].copy()
        y = df['attack_type']
        
        # --- NEW: Encode Attack Types (Target Variable) ---
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.target_encoder = target_encoder
        logger.info(f"Attack labels mapped to: {dict(zip(range(len(target_encoder.classes_)), target_encoder.classes_))}")
        
        # Split data
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # --- Categorical Encoding for Features (protocol_type) ---
        le = LabelEncoder()
        le.fit(X_train['protocol_type'])
        
        X_train['encoded_protocol_type'] = le.transform(X_train['protocol_type'])
        X_test['encoded_protocol_type'] = le.transform(X_test['protocol_type'])
        
        pipeline_features = NUM_FEATURES + ['encoded_protocol_type']
        
        # Create and train pipeline
        self.pipeline = self.create_pipeline()
        self.label_encoder = le
        
        logger.info(f"\nTraining Random Forest model on {len(X_train)} samples...")
        self.pipeline.fit(X_train[pipeline_features], y_train_encoded)
        
        # Evaluate model
        y_pred_encoded = self.pipeline.predict(X_test[pipeline_features])
        self.model_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        
        logger.info(f"✓ MODEL TRAINING COMPLETE! Accuracy: {self.model_accuracy*100:.2f}%")
        
        # Update global stats
        global stats
        stats['accuracy'] = round(self.model_accuracy * 100, 1)
        
        # Save model
        if save_model:
            model_data = {
                'pipeline': self.pipeline,
                'label_encoder': self.label_encoder,
                'target_encoder': self.target_encoder, # Save new encoder
                'accuracy': self.model_accuracy,
                'feature_names': ALL_FEATURES,
                'trained_at': datetime.now().isoformat()
            }
            joblib.dump(model_data, MODEL_PATH)
            logger.info(f"✓ Multi-class model saved to: {MODEL_PATH}")
        
        self.is_trained = True
        return self.model_accuracy
    
    def load_model(self):
        """Load trained model from file."""
        if os.path.exists(MODEL_PATH):
            try:
                model_data = joblib.load(MODEL_PATH)
                self.pipeline = model_data['pipeline']
                self.label_encoder = model_data['label_encoder']
                self.target_encoder = model_data['target_encoder'] # Load new encoder
                self.model_accuracy = model_data.get('accuracy', 0.0)
                self.is_trained = True
                
                global stats
                stats['accuracy'] = round(self.model_accuracy * 100, 1)
                
                logger.info(f"✓ Multi-class model loaded successfully. Accuracy: {self.model_accuracy*100:.2f}%")
                return True
            except Exception as e:
                logger.error(f"Error loading model, will try to train: {e}")
                return False
        return False
    
    def predict(self, input_data):
        """Make prediction on input data."""
        if not self.is_trained:
            if not self.load_model():
                self.train_model()

        try:
            df = pd.DataFrame([input_data])
            
            # Data Validation
            for feature in ALL_FEATURES:
                if feature not in df.columns:
                    raise ValueError(f"Missing required feature: {feature}")
            
            df_processed = df.copy()
            
            # Feature Encoding (protocol_type)
            protocol_type = df_processed['protocol_type'].iloc[0]
            if protocol_type not in self.label_encoder.classes_:
                protocol_type = self.label_encoder.classes_[0]
            
            df_processed['encoded_protocol_type'] = self.label_encoder.transform([protocol_type])
            
            pipeline_features = NUM_FEATURES + ['encoded_protocol_type']
            
            # Prediction on encoded target
            prediction_encoded = self.pipeline.predict(df_processed[pipeline_features])[0]
            prediction_proba = self.pipeline.predict_proba(df_processed[pipeline_features])[0]
            confidence = max(prediction_proba) * 100
            
            # --- NEW: Decode the prediction ---
            prediction = self.target_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get Feature Importance
            classifier = self.pipeline.named_steps['classifier']
            feature_importance = dict(zip(pipeline_features, classifier.feature_importances_))
            
            # Update global stats
            global stats
            stats['total_scans'] += 1
            if prediction != 'normal':
                stats['threats_detected'] += 1
            else:
                stats['normal_traffic'] += 1
            
            return {
                'prediction': prediction,
                'is_threat': prediction != 'normal',
                'confidence': round(confidence, 2),
                'feature_importance': {k: round(float(v), 5) for k, v in feature_importance.items()},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

# Initialize predictor
predictor = NIDSPredictor()

# --- HTML Template (UPDATED) ---
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Intrusion Detection System - Multi-Class Analytics</title>
    <style>
        /* Styles remain mostly the same, ensuring visual distinction for threats */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: #ffffff;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 20px;
            background: rgba(0, 255, 136, 0.05);
            border-radius: 20px;
            border: 1px solid rgba(0, 255, 136, 0.2);
        }
        .header h1 {
            font-size: 3em;
            background: linear-gradient(135deg, #00ff88, #0088ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .train-btn {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            margin-top: 15px;
        }
        .cyber-card {
            background: rgba(0, 20, 40, 0.8);
            border: 1px solid rgba(0, 255, 136, 0.2);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        }
        .form-group { margin-bottom: 15px; }
        .form-label {
            color: #00ff88;
            font-weight: 600;
            display: block;
            margin-bottom: 5px;
        }
        .form-input {
            width: 100%;
            background: rgba(0, 40, 80, 0.6);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 8px;
            padding: 10px;
            color: #ffffff;
        }
        .predict-btn {
            background: linear-gradient(135deg, #00ff88, #0088ff);
            color: #000;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-weight: 700;
            cursor: pointer;
            width: 100%;
        }
        .result-display {
            background: rgba(0, 20, 40, 0.9);
            border: 2px solid rgba(0, 255, 136, 0.3);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            min-height: 150px;
        }
        .result-normal { border-color: #00ff88; }
        .result-threat { 
            border-color: #ff0088; 
            background: rgba(255, 0, 136, 0.1); 
        }
        .result-threat .attack-type {
            font-size: 2em;
            color: #ff0088;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px; }
        .stat-item {
            background: rgba(0, 40, 80, 0.6);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .stat-number { font-size: 2em; color: #00ff88; }
        .stat-label { color: #a0a0a0; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NETWORK INTRUSION DETECTOR (MULTI-CLASS)</h1>
            <p>Model Classifies Specific Attack Types (DoS, R2L, etc.)</p>
            <button class="train-btn" onclick="trainModel()">Retrain Model</button>
        </div>
        
        <div class="cyber-card">
            <h2 style="color: #00ff88;">Enter Network Flow Data</h2>
            <form id="predictionForm" onsubmit="predictTraffic(event)">
                <div class="form-group">
                    <label class="form-label">Duration (seconds)</label>
                    <input type="number" class="form-input" name="duration" step="0.01" value="0.5" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Protocol Type</label>
                    <select class="form-input" name="protocol_type" required>
                        <option value="tcp">TCP</option>
                        <option value="udp">UDP</option>
                        <option value="icmp">ICMP</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Source Bytes</label>
                    <input type="number" class="form-input" name="src_bytes" value="5000" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Destination Bytes</label>
                    <input type="number" class="form-input" name="dst_bytes" value="2000" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Failed Logins</label>
                    <input type="number" class="form-input" name="num_failed_logins" value="0" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Connection Count</label>
                    <input type="number" class="form-input" name="count" value="10" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Service Count</label>
                    <input type="number" class="form-input" name="srv_count" value="5" required>
                </div>
                <div class="form-group">
                    <label class="form-label">Same Service Rate (0-1)</label>
                    <input type="number" class="form-input" name="same_srv_rate" step="0.01" min="0" max="1" value="0.9" required>
                </div>
                <button type="submit" class="predict-btn">ANALYZE TRAFFIC</button>
            </form>
        </div>
        
        <div class="cyber-card">
            <div class="result-display" id="resultDisplay">
                <div style="font-size: 3em;">⚡</div>
                <div style="font-size: 1.3em; margin: 10px 0;">Ready for Analysis</div>
            </div>
        </div>
        
        <div class="cyber-card">
            <h2 style="color: #00ff88;">System Statistics</h2>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-number" id="totalScans">{{ stats.total_scans }}</div>
                    <div class="stat-label">Total Scans</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="threats">{{ stats.threats_detected }}</div>
                    <div class="stat-label">Threats Detected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="normal">{{ stats.normal_traffic }}</div>
                    <div class="stat-label">Normal Traffic</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="accuracy">{{ "%.1f" | format(stats.accuracy) }}%</div>
                    <div class="stat-label">Accuracy (Multi-Class)</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function predictTraffic(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            // Convert numbers
            ['duration', 'src_bytes', 'dst_bytes', 'num_failed_logins', 'count', 'srv_count', 'same_srv_rate'].forEach(field => {
                data[field] = parseFloat(data[field]) || 0;
            });
            
            const resultDiv = document.getElementById('resultDisplay');
            resultDiv.innerHTML = '<div style="font-size: 2em;">⏳</div><div>Analyzing...</div>';
            resultDiv.className = 'result-display'; // Reset class
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.status !== 200) {
                     throw new Error(result.error || 'Prediction failed');
                }
                
                const isThreat = result.is_threat;
                const predictionText = result.prediction.toUpperCase().replace(/_/g, ' '); // Format label
                
                resultDiv.className = `result-display ${isThreat ? 'result-threat' : 'result-normal'}`;

                if (isThreat) {
                    resultDiv.innerHTML = `
                        <div style="font-size: 3em;">🚨</div>
                        <div class="attack-type">${predictionText}</div>
                        <div style="font-size: 1.1em; margin: 5px 0;">ATTACK DETECTED!</div>
                        <div>Confidence: ${result.confidence}%</div>
                        <div style="font-size: 0.8em; margin-top: 10px; color: #aaa;">(${result.timestamp.substring(11, 19)})</div>
                    `;
                } else {
                     resultDiv.innerHTML = `
                        <div style="font-size: 3em;">✅</div>
                        <div style="font-size: 1.3em; margin: 10px 0;">NORMAL TRAFFIC</div>
                        <div>Confidence: ${result.confidence}%</div>
                        <div style="font-size: 0.8em; margin-top: 10px; color: #aaa;">(${result.timestamp.substring(11, 19)})</div>
                    `;
                }
                
                // Update stats
                const statsResp = await fetch('/api/status');
                const stats = await statsResp.json();
                document.getElementById('totalScans').textContent = stats.total_scans;
                document.getElementById('threats').textContent = stats.threats_detected;
                document.getElementById('normal').textContent = stats.normal_traffic;
                
            } catch (error) {
                resultDiv.className = 'result-display';
                resultDiv.innerHTML = '<div style="font-size: 2em;">❌</div><div>Error: ' + error.message + '</div>';
            }
        }
        
        async function trainModel() {
            if (!confirm('This will retrain the MULTI-CLASS model. Continue?')) return;
            
            try {
                const trainBtn = document.querySelector('.train-btn');
                trainBtn.textContent = 'Training...';
                trainBtn.disabled = true;
                
                const response = await fetch('/train', {method: 'POST'});
                const result = await response.json();
                
                if (response.status !== 200) {
                     throw new Error(result.error || 'Training failed');
                }
                
                alert(`Multi-Class Model trained successfully! Accuracy: ${(result.accuracy * 100).toFixed(2)}%`);
                location.reload();
            } catch (error) {
                alert('Training failed: ' + error.message);
                document.querySelector('.train-btn').textContent = 'Retrain Model';
                document.querySelector('.train-btn').disabled = false;
            }
        }
    </script>
</body>
</html>'''

# --- Flask API Endpoints (No change needed) ---

# All API endpoints remain the same, relying on the updated NIDSPredictor class.
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, stats=stats)

@app.route("/train", methods=["POST"])
def train():
    try:
        accuracy = predictor.train_model()
        return jsonify({"success": True, "accuracy": accuracy})
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result = predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/status")
def status():
    global stats
    return jsonify({
        "model_trained": predictor.is_trained,
        "model_accuracy": round(predictor.model_accuracy * 100, 1),
        "total_scans": stats['total_scans'],
        "threats_detected": stats['threats_detected'],
        "normal_traffic": stats['normal_traffic']
    })

# --- Main Execution ---
if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🛡️  NETWORK INTRUSION DETECTION SYSTEM (MULTI-CLASS) - Flask Server")
    print("="*70)
    
    try:
        if not predictor.load_model():
            print("\n📊 No existing model found. Attempting to train multi-class model...")
            predictor.train_model(save_model=True)
            print("Training complete.")
    except FileNotFoundError as e:
        print("\n" + "!"*70)
        print(f"FATAL ERROR: {e}")
        print(f"Please confirm 'network_intrusions (1).csv' is in the 'data/' directory.")
        print("!"*70)
    except Exception as e:
        print(f"\nFATAL ERROR during model initialization or training: {e}")

    print(f"\n📡 Starting server at http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')