from pathlib import Path
import joblib
import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"

FEATURE_ORDER = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

def load_artifacts(model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
    """Load trained model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def _vectorize(payload: dict) -> np.ndarray:
    """Convert dict to 2D numpy array in the correct feature order."""
    return np.array([[payload[f] for f in FEATURE_ORDER]])

def predict(payload: dict, model, scaler) -> float:
    """Make a single prediction from a JSON-like dict."""
    X = _vectorize(payload)
    Xs = scaler.transform(X)
    y = model.predict(Xs)
    return float(y[0])