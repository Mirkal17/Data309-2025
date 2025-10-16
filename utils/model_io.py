import os
import gdown

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def download_file_from_gdrive(file_id: str, save_path: str) -> str:
    """Download file from Google Drive using its ID."""
    if os.path.exists(save_path):
        print(f"✅ Using cached model: {save_path}")
        return save_path
    ensure_dir(os.path.dirname(save_path))
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇️ Downloading model from Google Drive -> {save_path}")
    gdown.download(url, save_path, quiet=False)
    return save_path
