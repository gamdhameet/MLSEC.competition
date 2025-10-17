import os
import envparse
from defender.apps import create_app

# Advanced LightGBM Model with Enhanced Features
from defender.models.advanced_model import AdvancedMalwareModel

if __name__ == "__main__":
    # Get threshold from environment variable (default: 0.80 optimized for FPR < 1%)
    model_thresh = envparse.env("DF_MODEL_THRESH", cast=float, default=0.80)
    
    # Load Advanced LightGBM model
    adv_model_path = os.path.join(os.path.dirname(__file__), "models/advanced_model.pkl")
    model = AdvancedMalwareModel(
        model_path=adv_model_path, 
        thresh=model_thresh, 
        name="Advanced-LightGBM-Detector"
    )

    app = create_app(model)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 8080

    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()

    # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"
