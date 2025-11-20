from pathlib import Path

GDRIVE_MOUNT = Path.home() / "gdrive"
RAW_DATA_DIR = GDRIVE_MOUNT / "DepressionDetectionData" / "raw"
PROCESSED_DIR = GDRIVE_MOUNT / "DepressionDetectionData" / "processed"

DOWNLOAD_CONFIG = {
    "RAVDESS_AUDIO": True,
    "RAVDESS_VIDEO_ACTORS": [],
    "MELD": True,
    "FER2013": True,
    "TESS": True,
    "DEPRESSION_REDDIT": True,
    "MENTAL_HEALTH_SURVEY": True,
    "SENTIMENT140": False,
    "MENTAL_HEALTH_SOCIAL": False
}

PREPROCESS_CONFIG = {
    "RAVDESS_AUDIO": True,
    "MELD": True,
    "FER2013": True,
    "TESS": True,
    "DEPRESSION_REDDIT": True,
    "MENTAL_HEALTH_SURVEY": True
}

SAMPLE_LIMITS = {
    "RAVDESS_MAX_FILES": 0,
    "TESS_MAX_FILES": 0,
    "MELD_MAX_FILES": 0,
    "REDDIT_MAX_ROWS": 0,
    "SENTIMENT140_ROWS": 200000,
    "MENTAL_HEALTH_SOCIAL_ROWS": 50000
}

AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_mfcc": 13,
    "n_fft": 2048,
    "hop_length": 512
}

MELD_FILE_IDS = {
    "train": "1hDmpPDPf9mXsFN1PRXXqaXfXiQ-K4Dq0",
    "dev": "1Yl5B6jXnLd7QVpCuLd4qqjK7ZSCVzSWW",
    "test": "1sG4evqCfz5BwOQBNVQpPGW2X1D9gQQHb"
}