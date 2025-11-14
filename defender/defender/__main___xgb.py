import os
import envparse
from defender.apps import create_app

# XGBoost Model
from defender.models.xgboost_model import XGBoostMalwareModel

if __name__ == "__main__":
    # Get threshold from environment variable (will be loaded from pickle if available)
    model_thresh = envparse.env("DF_MODEL_THRESH", cast=float, default=0.5)
    
    # Load XGBoost model
    xgb_model_path = os.path.join(os.path.dirname(__file__), "models/vkota_xgb_model.json")
    model = XGBoostMalwareModel(
        model_path=xgb_model_path, 
        thresh=model_thresh, 
        name="XGBoost-Detector"
    )

    app = create_app(model)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 8080

    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()

    # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"

