import os
import envparse
from defender.apps import create_app
import pickle

# CUSTOMIZE: import model to be used
# from defender.models.ember_model import StatefulNNEmberModel
# from defender.models.nfs_behemot_model import NFSBehemotModel
# from defender.models.nfs_model import PEAttributeExtractor, NFSModel, NeedForSpeedModel
# from defender.models.simple_bert_model import SimpleBERTMalwareModel
# from defender.models.reliable_nn_model import ReliableNNMalwareModel

# NEW IMPORTS for your model
from defender import dropper_aware_model as dam
from defender.dropper_aware_model import DropperAwareModel

if __name__ == "__main__":
    # retrive config values from environment variables
    model_gz_path = envparse.env("DF_MODEL_GZ_PATH", cast=str, default="models/ember_model.txt.gz")
    model_thresh = envparse.env("DF_MODEL_THRESH", cast=float, default=0.8336)
    model_name = envparse.env("DF_MODEL_NAME", cast=str, default="ember")
    model_ball_thresh = envparse.env("DF_MODEL_BALL_THRESH", cast=float, default=0.25)
    model_max_history = envparse.env("DF_MODEL_HISTORY", cast=int, default=10_000)

    # construct absolute path to ensure the correct model is loaded
    if not model_gz_path.startswith(os.sep):
        model_gz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_gz_path)

    # CUSTOMIZE: app and model instance
    # model = StatefulNNEmberModel(model_gz_path,
    #                              model_thresh,
    #                              model_ball_thresh,
    #                              model_max_history,
    #                              model_name)
    
    # model = NFSBehemotModel()
    # model = NFSModel(open(os.path.dirname(__file__) + "/models/nfs_full.pickle", "rb"))
    # model = NFSModel(open(os.path.dirname(__file__) + "/models/nfs_libraries_functions_nostrings.pickle", "rb"))
    
    # Use Reliable Neural Network malware detection model
    # reliable_nn_model_path = os.path.join(os.path.dirname(__file__), "models/reliable_nn_model.pkl")
    # Use a much lower threshold to reduce false negatives - 0.3 instead of 0.8336
    # model = ReliableNNMalwareModel(model_path=reliable_nn_model_path, thresh=0.3, name="Reliable-NN-Malware-Detector")

    print("[INFO] Loading Dropper-Aware model from pickle...")

    model_path = os.environ.get("MODEL_PATH",
                                os.path.join(os.path.dirname(__file__), "dropper_model.pkl"))

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Pickle model not found at: {model_path}")

    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)

    dam._scaler = model_dict.get("scaler")
    dam._clf = model_dict.get("classifier")
    dam._threshold = float(model_dict.get("threshold", 0.5))
    dam._loaded = True

    model = DropperAwareModel()
    print(f"[OK] Dropper-Aware model loaded successfully from {model_path}")

    #Run app as normal
    app = create_app(model)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 8080

    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()

    # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"
