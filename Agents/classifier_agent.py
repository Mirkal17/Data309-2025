# Agents/classifier_agent.py
import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# --- Ensure NLTK stopwords available ---
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

STEMMER = SnowballStemmer("english")

# ---------- Google Drive one-file helper (no external utils) ----------
# Requires: pip install gdown ; and set CLASSIFIER_MODEL_ID in .env
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _download_from_gdrive(file_id: str, save_path: str):
    """Download a single file from Google Drive by ID if not present."""
    if os.path.exists(save_path):
        print(f"✅ Using cached model: {save_path}")
        return
    if not file_id:
        raise RuntimeError(
            "CLASSIFIER_MODEL_ID is not set. Put it in your .env or export as an env var."
        )
    import gdown  # local import so module works even if gdown not installed elsewhere
    _ensure_dir(os.path.dirname(save_path))
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇️ Downloading classifier from Google Drive -> {save_path}")
    gdown.download(url, save_path, quiet=False)

# ---------- Paths (absolute so Streamlit cwd doesn't matter) ----------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(HERE, "..", "trained_models"))

RF_PATH  = os.path.join(MODEL_DIR, "random_forest_model.pkl")   # BIG file (Drive)
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")             # already in repo
NZV_PATH = os.path.join(MODEL_DIR, "variance_threshold.pkl")     # already in repo

# Try to load .env if present (harmless if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

CLASSIFIER_MODEL_ID = os.getenv("CLASSIFIER_MODEL_ID")  # <-- put your Drive file ID here (in .env)

# Download RF model if not present
if not os.path.exists(RF_PATH):
    _download_from_gdrive(CLASSIFIER_MODEL_ID, RF_PATH)

# --- Load models (vectorizer & nzv assumed present locally) ---
print("🔹 Loading classifier components...")
rf  = joblib.load(RF_PATH)
vectorizer = joblib.load(VEC_PATH)
nzv = joblib.load(NZV_PATH)
print("✅ Classifier ready.")

# ---------- Text cleaning & inference ----------
def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [STEMMER.stem(w) for w in tokens]
    return " ".join(tokens)

def ticket_classification(ticket_description: str) -> str:
    """Return predicted department/category for a support ticket."""
    cleaned = _clean_text(ticket_description)
    X = vectorizer.transform([cleaned])
    X = nzv.transform(X)
    pred = rf.predict(X)
    return pred[0]
