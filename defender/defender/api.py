"""
HTTP API for malware detection
"""
from flask import Flask, request, jsonify
import logging
import os
from .models.predictor import MalwarePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize predictor
MODEL_DIR = os.getenv('DF_MODEL_DIR', '/opt/defender/defender/models')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'random_forest_scaler.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'random_forest_features.pkl')
THRESHOLD = float(os.getenv('DF_MODEL_THRESH', '0.6'))

logger.info(f"Initializing predictor from {MODEL_DIR}")
predictor = MalwarePredictor(MODEL_PATH, SCALER_PATH, FEATURES_PATH, threshold=THRESHOLD)
logger.info("Predictor initialized successfully")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'random_forest', 'threshold': THRESHOLD})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict malware from uploaded PE file
    
    Request: POST with file in 'file' field
    Response: JSON with prediction
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read file bytes
        file_bytes = file.read()
        
        if len(file_bytes) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        logger.info(f"Processing file: {file.filename} ({len(file_bytes)} bytes)")
        
        # Predict
        result = predictor.predict(file_bytes)
        result['filename'] = file.filename
        
        logger.info(f"Prediction: {result['label']} (prob={result['probability']:.4f})")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)