import os
import envparse
from defender.apps import create_app

# Decision Tree Model
from defender.models.decision_tree_model import DecisionTreeMalwareModel

if __name__ == "__main__":
    # Get threshold from environment variable (will be loaded from pickle if available)
    model_thresh = envparse.env("DF_MODEL_THRESH", cast=float, default=0.5)
    
    # Load Decision Tree model
    dt_model_path = os.path.join(os.path.dirname(__file__), "models/dt_single_aggressive.pkl")
    model = DecisionTreeMalwareModel(
        model_path=dt_model_path, 
        thresh=model_thresh, 
        name="Decision-Tree-Detector"
    )

    app = create_app(model)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 8080

    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()

    # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"

