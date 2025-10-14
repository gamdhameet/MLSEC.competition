import os
import envparse
from defender.apps import create_app

# Random Forest Model Only
from defender.models.random_forest_model import RandomForestMalwareModel

if __name__ == "__main__":
    # Get threshold from environment variable (default: 0.21 optimized threshold)
    model_thresh = envparse.env("DF_MODEL_THRESH", cast=float, default=0.21)
    
    # Load Random Forest model
    rf_model_path = os.path.join(os.path.dirname(__file__), "models/random_forest_model.pkl")
    model = RandomForestMalwareModel(
        model_path=rf_model_path, 
        thresh=model_thresh, 
        name="Random-Forest-Detector"
    )

    app = create_app(model)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 8080

    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()

    # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"
