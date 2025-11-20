<think>
**Compiling complete step-by-step instructions**

I'm putting together a thorough, friendly guide that walks through creating the repo, adding all necessary files, pushing to GitHub, launching Codespaces, setting up Kaggle credentials, configuring rclone for Drive mounting, running download and preprocess scripts, and verifying outputs—all with code snippets and clear steps to make it easy to follow.
</think>Here’s the full, self-contained blueprint to get your entire depression-detection dataset pipeline running inside **GitHub Codespaces**, downloading directly into Google Drive, and saving all preprocessed outputs back into Drive—no local storage required.

---

## 🧭 Overview
You’ll set up a repo with the pipeline code, spin up a Codespace (cloud dev environment), mount Google Drive via `rclone`, run the downloader, then run the preprocessing script. Everything will land neatly in Drive.

---

## ✅ Step 1: Create the Repository Structure

```text
depression-detection-data/
├── .devcontainer/
│   └── devcontainer.json
├── .gitignore
├── README.md
├── requirements.txt
├── config.py
├── download_datasets.py
└── preprocess_datasets.py
```

---

## 📄 Step 2: Populate Files

### `.devcontainer/devcontainer.json`
```json
{
  "name": "Depression Detection Data Pipeline",
  "image": "mcr.microsoft.com/devcontainers/python:3.10-bullseye",
  "features": {
    "ghcr.io/devcontainers-contrib/features/rclone:1": {}
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.black-formatter",
        "ms-toolsai.jupyter"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.formatting.provider": "black",
        "editor.formatOnSave": true
      }
    }
  },
  "postCreateCommand": "pip install --upgrade pip && pip install -r requirements.txt",
  "remoteUser": "vscode",
  "mounts": [
    "source=${localEnv:HOME}/.kaggle,target=/home/vscode/.kaggle,type=bind,consistency=cached"
  ]
}
```

### `.gitignore`
```gitignore
__pycache__/
*.py[cod]
*$py.class
env/
venv/
*.egg-info/
dist/
build/
.ipynb_checkpoints/
*.ipynb
raw/
processed/
*.zip
*.tar.gz
*.csv
*.npz
*.parquet
rclone.conf
.kaggle/
kaggle.json
.vscode/
.idea/
.DS_Store
Thumbs.db
```

### `requirements.txt`
```text
gdown>=4.7.1
kaggle>=1.5.16
librosa>=0.10.0
soundfile>=0.12.1
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
pyarrow>=12.0.0
clean-text==0.6.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### `config.py`
```python
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
```

### `download_datasets.py`
```python
#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
from typing import List
import gdown

from config import (
    GDRIVE_MOUNT, RAW_DATA_DIR, DOWNLOAD_CONFIG, MELD_FILE_IDS
)

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def check_mount():
    if not GDRIVE_MOUNT.exists():
        print(f"""
❌ Google Drive not mounted at {GDRIVE_MOUNT}

Run these commands inside Codespaces:
    mkdir -p {GDRIVE_MOUNT}
    rclone mount gdrive: {GDRIVE_MOUNT} --vfs-cache-mode writes --daemon

Then rerun this script.
""")
        sys.exit(1)
    print(f"✅ Google Drive mounted at {GDRIVE_MOUNT}")

def run_cmd(cmd: List[str], desc: str = ""):
    print(f"\n👉 {desc or 'Running command'}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed: {exc}")

def check_kaggle():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("""
⚠️ Kaggle API not found.

In Codespaces terminal:
    mkdir -p ~/.kaggle
Upload kaggle.json (from kaggle.com/account) via VS Code explorer.
    chmod 600 ~/.kaggle/kaggle.json

Then rerun this script.
""")
        return False
    kaggle_json.chmod(0o600)
    print("✅ Kaggle API ready")
    return True

def download_gdrive(file_id: str, dest: Path, desc: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n⬇️ {desc}")
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(dest), quiet=False)
    except Exception:
        gdown.download(url, str(dest), quiet=False, fuzzy=True)

def download_ravdess():
    print("\n=== RAVDESS ===")
    ravdess_dir = RAW_DATA_DIR / "ravdess"
    ravdess_dir.mkdir(exist_ok=True)
    if DOWNLOAD_CONFIG["RAVDESS_AUDIO"]:
        audio_zip = ravdess_dir / "Audio_Speech_Actors_01-24.zip"
        if not audio_zip.exists():
            run_cmd(
                ["wget", "-c", "-O", str(audio_zip),
                 "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"],
                "Downloading RAVDESS audio (2.4 GB)"
            )
    if DOWNLOAD_CONFIG["RAVDESS_VIDEO_ACTORS"]:
        for actor in DOWNLOAD_CONFIG["RAVDESS_VIDEO_ACTORS"]:
            actor_zip = ravdess_dir / f"Video_Speech_Actor_{actor:02d}.zip"
            if not actor_zip.exists():
                run_cmd(
                    ["wget", "-c", "-O", str(actor_zip),
                     f"https://zenodo.org/record/1188976/files/Video_Speech_Actor_{actor:02d}.zip"],
                    f"Downloading RAVDESS video actor {actor:02d}"
                )

def download_meld():
    print("\n=== MELD ===")
    meld_dir = RAW_DATA_DIR / "meld"
    meld_dir.mkdir(exist_ok=True)
    for split, file_id in MELD_FILE_IDS.items():
        archive = meld_dir / f"{split}.tar.gz"
        if not archive.exists():
            download_gdrive(file_id, archive, f"MELD {split}")

def download_kaggle_dataset(slug: str, target: Path, desc: str):
    if not check_kaggle():
        print(f"⏭️ Skipping {desc} (Kaggle missing)")
        return False
    target.mkdir(exist_ok=True)
    run_cmd(["kaggle", "datasets", "download", "-d", slug, "-p", str(target)], desc)
    return True

def download_fer2013():
    print("\n=== FER-2013 ===")
    download_kaggle_dataset("msambare/fer2013", RAW_DATA_DIR / "fer2013", "Downloading FER-2013")

def download_tess():
    print("\n=== TESS ===")
    download_kaggle_dataset("ejlok1/toronto-emotional-speech-set-tess",
                            RAW_DATA_DIR / "tess", "Downloading TESS")

def download_depression_reddit():
    print("\n=== Depression Reddit ===")
    download_kaggle_dataset("infamouscoder/depression-reddit-cleaned",
                            RAW_DATA_DIR / "depression_reddit",
                            "Downloading Depression Reddit")

def download_mental_health_survey():
    print("\n=== Mental Health Survey ===")
    download_kaggle_dataset("osmi/mental-health-in-tech-survey",
                            RAW_DATA_DIR / "mental_health_survey",
                            "Downloading Mental Health Survey")

def main():
    print("="*60)
    print("🚀 Dataset Downloader (Codespaces → Google Drive)")
    print("="*60)
    check_mount()

    if DOWNLOAD_CONFIG["RAVDESS_AUDIO"] or DOWNLOAD_CONFIG["RAVDESS_VIDEO_ACTORS"]:
        download_ravdess()
    if DOWNLOAD_CONFIG["MELD"]:
        download_meld()
    if DOWNLOAD_CONFIG["FER2013"]:
        download_fer2013()
    if DOWNLOAD_CONFIG["TESS"]:
        download_tess()
    if DOWNLOAD_CONFIG["DEPRESSION_REDDIT"]:
        download_depression_reddit()
    if DOWNLOAD_CONFIG["MENTAL_HEALTH_SURVEY"]:
        download_mental_health_survey()

    print("\n🎉 Downloads saved under:", RAW_DATA_DIR)

if __name__ == "__main__":
    main()
```

### `preprocess_datasets.py`
```python
#!/usr/bin/env python3
import sys
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from cleantext import clean
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from config import (
    GDRIVE_MOUNT, RAW_DATA_DIR, PROCESSED_DIR,
    PREPROCESS_CONFIG, AUDIO_CONFIG, SAMPLE_LIMITS
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def check_mount():
    if not GDRIVE_MOUNT.exists():
        print(f"❌ Google Drive not mounted at {GDRIVE_MOUNT}")
        sys.exit(1)
    print(f"✅ Google Drive mounted at {GDRIVE_MOUNT}")

def extract_zip(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)

def extract_tar(tar_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:*') as tf:
        tf.extractall(target_dir)

def audio_features(wav: Path):
    try:
        y, sr = librosa.load(wav, sr=AUDIO_CONFIG["sample_rate"])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=AUDIO_CONFIG["n_mfcc"])
        zcr = librosa.feature.zero_crossing_rate(y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)

        feats = {
            "duration_sec": len(y) / sr,
            "zcr_mean": float(zcr.mean()),
            "zcr_std": float(zcr.std()),
            "spec_centroid_mean": float(cent.mean()),
            "spec_centroid_std": float(cent.std()),
            "spec_bandwidth_mean": float(bw.mean()),
            "spec_rolloff_mean": float(rolloff.mean()),
            "rms_mean": float(rms.mean()),
            "rms_std": float(rms.std())
        }
        for i in range(AUDIO_CONFIG["n_mfcc"]):
            feats[f"mfcc_{i+1}_mean"] = float(mfcc[i].mean())
            feats[f"mfcc_{i+1}_std"] = float(mfcc[i].std())
        return feats
    except Exception as exc:
        print(f"⚠️ {wav.name}: {exc}")
        return None

def clean_text_basic(text: str) -> str:
    return clean(
        str(text), fix_unicode=True, to_ascii=False, lower=True,
        no_line_breaks=True, no_urls=True, no_emails=True,
        no_phone_numbers=True, no_currency_symbols=True
    ).strip()

def featureize_text(df: pd.DataFrame, col: str):
    df[f"{col}_clean"] = df[col].fillna("").apply(clean_text_basic)
    df[f"{col}_char_len"] = df[f"{col}_clean"].str.len()
    df[f"{col}_word_len"] = df[f"{col}_clean"].str.split().str.len()
    df[f"{col}_avg_word_len"] = df[f"{col}_clean"].apply(
        lambda s: np.mean([len(w) for w in s.split()]) if s else 0
    )
    df[f"{col}_exclaim_count"] = df[f"{col}_clean"].str.count("!")
    df[f"{col}_question_count"] = df[f"{col}_clean"].str.count(r"\?")
    df[f"{col}_uppercase_ratio"] = df[col].apply(
        lambda s: sum(c.isupper() for c in str(s)) / len(str(s))
        if len(str(s)) else 0
    )
    return df

def preprocess_ravdess():
    print("\n=== Preprocess RAVDESS ===")
    zip_path = RAW_DATA_DIR / "ravdess" / "Audio_Speech_Actors_01-24.zip"
    if not zip_path.exists():
        print("⏭️ Missing RAVDESS audio")
        return
    extract_dir = RAW_DATA_DIR / "ravdess" / "extracted"
    if not extract_dir.exists():
        extract_zip(zip_path, extract_dir)
    audio_root = extract_dir / "Audio_Speech_Actors_01-24"
    files = sorted(audio_root.rglob("*.wav"))
    if SAMPLE_LIMITS["RAVDESS_MAX_FILES"] > 0:
        files = files[:SAMPLE_LIMITS["RAVDESS_MAX_FILES"]]
    emotion_map = {
        "01": "neutral","02": "calm","03": "happy","04": "sad",
        "05": "angry","06": "fearful","07": "disgust","08": "surprised"
    }
    rows = []
    for wav in tqdm(files, desc="Audio features"):
        parts = wav.stem.split('-')
        meta = {
            "filename": wav.name,
            "modality": parts[0],
            "vocal_channel": parts[1],
            "emotion_id": parts[2],
            "emotion": emotion_map.get(parts[2], "unknown"),
            "intensity": "normal" if parts[3] == "01" else "strong",
            "statement": parts[4],
            "repetition": parts[5],
            "actor_id": int(parts[6]),
            "gender": "female" if int(parts[6]) % 2 == 0 else "male"
        }
        feats = audio_features(wav)
        if feats:
            rows.append({**meta, **feats})
    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "ravdess_audio_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {len(df)} rows")

def preprocess_meld():
    print("\n=== Preprocess MELD ===")
    meld_dir = RAW_DATA_DIR / "meld"
    extract_dir = meld_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    for split in ["train", "dev", "test"]:
        tar_path = meld_dir / f"{split}.tar.gz"
        if tar_path.exists() and not (extract_dir / split).exists():
            extract_tar(tar_path, extract_dir)
    csv_files = list(extract_dir.rglob("*_sent_emo.csv"))
    if not csv_files:
        print("⏭️ MELD CSVs missing")
        return
    frames = []
    for csv in csv_files:
        split = csv.stem.split("_")[0]
        df = pd.read_csv(csv)
        if "Utterance" not in df.columns:
            continue
        df = df.rename(columns={"Utterance": "text"})
        df = featureize_text(df, "text")
        df["split"] = split
        frames.append(df[[
            "Dialogue_ID","Utterance_ID","Speaker","Season","Episode",
            "Emotion","Sentiment","text","text_clean","text_char_len",
            "text_word_len","text_avg_word_len","text_exclaim_count",
            "text_question_count","text_uppercase_ratio","split"
        ]])
    meld_df = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "meld_text_features.parquet"
    meld_df.to_parquet(out_path, index=False)
    print(f"✅ Saved {len(meld_df)} rows")

def preprocess_fer2013():
    print("\n=== Preprocess FER-2013 ===")
    fer_dir = RAW_DATA_DIR / "fer2013"
    zip_path = fer_dir / "fer2013.zip"
    if not zip_path.exists():
        print("⏭️ Missing FER-2013 zip")
        return
    extract_zip(zip_path, fer_dir)
    csv_path = fer_dir / "fer2013.csv"
    df = pd.read_csv(csv_path)
    pixels = np.vstack(
        df["pixels"].apply(lambda row: np.fromstring(row, sep=" ", dtype=np.uint8))
    ).reshape(-1, 48, 48, 1) / 255.0
    np.savez_compressed(
        PROCESSED_DIR / "fer2013_images.npz",
        pixels=pixels, emotion=df["emotion"].values, usage=df["Usage"].values
    )
    metadata = pd.DataFrame({
        "emotion": df["emotion"],
        "emotion_label": df["emotion"].map({
            0:"angry",1:"disgust",2:"fear",3:"happy",
            4:"sad",5:"surprise",6:"neutral"
        }),
        "usage": df["Usage"]
    })
    metadata.to_parquet(PROCESSED_DIR / "fer2013_metadata.parquet", index=False)
    print(f"✅ Saved {len(df)} images + metadata")

def preprocess_tess():
    print("\n=== Preprocess TESS ===")
    tess_dir = RAW_DATA_DIR / "tess"
    zip_path = tess_dir / "toronto-emotional-speech-set-tess.zip"
    if not zip_path.exists():
        print("⏭️ TESS zip missing")
        return
    extract_dir = tess_dir / "extracted"
    if not extract_dir.exists():
        extract_zip(zip_path, extract_dir)
    root = extract_dir / "TESS Toronto emotional speech set data"
    wav_files = sorted(root.rglob("*.wav"))
    if SAMPLE_LIMITS["TESS_MAX_FILES"] > 0:
        wav_files = wav_files[:SAMPLE_LIMITS["TESS_MAX_FILES"]]
    rows = []
    for wav in tqdm(wav_files, desc="Audio features"):
        parts = wav.parent.name.split("_")
        meta = {
            "filename": wav.name,
            "speaker": parts[0],
            "emotion": parts[1].lower() if len(parts) > 1 else "unknown"
        }
        feats = audio_features(wav)
        if feats:
            rows.append({**meta, **feats})
    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "tess_audio_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {len(df)} rows")

def preprocess_depression_reddit():
    print("\n=== Preprocess Depression Reddit ===")
    reddit_dir = RAW_DATA_DIR / "depression_reddit"
    zip_path = reddit_dir / "depression-reddit-cleaned.zip"
    if not zip_path.exists():
        print("⏭️ Missing Reddit zip")
        return
    extract_zip(zip_path, reddit_dir)
    csv_files = list(reddit_dir.glob("*.csv"))
    if not csv_files:
        print("⚠️ No CSV found")
        return
    df = pd.read_csv(csv_files[0])
    df.columns = df.columns.str.lower().str.strip()
    text_col = next((c for c in ["clean_text","text","post","content"] if c in df.columns), None)
    if not text_col:
        print("⚠️ No text column detected")
        return
    df = df.rename(columns={text_col: "post"})
    df = featureize_text(df, "post")
    if "depression" in df.columns:
        df["label"] = df["depression"].map({0: "control", 1: "depression"})
    out_path = PROCESSED_DIR / "depression_reddit_text.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {len(df)} rows")

def preprocess_mental_health_survey():
    print("\n=== Preprocess Mental Health Survey ===")
    survey_dir = RAW_DATA_DIR / "mental_health_survey"
    zip_path = survey_dir / "mental-health-in-tech-survey.zip"
    if not zip_path.exists():
        print("⏭️ Missing survey zip")
        return
    extract_zip(zip_path, survey_dir)
    csv_path = survey_dir / "survey.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").clip(18, 75)
    y = df["treatment"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    X = df.drop(columns=[c for c in ["treatment","Timestamp","comments","state","Country"] if c in df.columns])
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    pipeline = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical)
    ])
    X_processed = pipeline.fit_transform(X)
    cat_features = pipeline.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical)
    features = pd.DataFrame(X_processed, columns=list(numeric) + list(cat_features))
    features["treatment"] = y.values
    out_path = PROCESSED_DIR / "mental_health_survey.parquet"
    features.to_parquet(out_path, index=False)
    joblib.dump(pipeline, PROCESSED_DIR / "mental_health_preprocessor.joblib")
    print(f"✅ Saved {len(features)} rows + preprocessor")

def main():
    print("="*60)
    print("🔬 Preprocessing Pipeline")
    print("="*60)
    check_mount()
    if PREPROCESS_CONFIG["RAVDESS_AUDIO"]:
        preprocess_ravdess()
    if PREPROCESS_CONFIG["MELD"]:
        preprocess_meld()
    if PREPROCESS_CONFIG["FER2013"]:
        preprocess_fer2013()
    if PREPROCESS_CONFIG["TESS"]:
        preprocess_tess()
    if PREPROCESS_CONFIG["DEPRESSION_REDDIT"]:
        preprocess_depression_reddit()
    if PREPROCESS_CONFIG["MENTAL_HEALTH_SURVEY"]:
        preprocess_mental_health_survey()
    print("\n🎉 All outputs stored in:", PROCESSED_DIR)

if __name__ == "__main__":
    main()
```

### `README.md`
```markdown
# Depression Detection Data Pipeline (GitHub Codespaces + Google Drive)

## Quick Start
1. Fork or clone this repo.
2. Go to **Code → Codespaces → Create codespace**.
3. Wait for the devcontainer to build (Python + rclone installed).
4. **Mount Google Drive:**
   ```bash
   mkdir -p ~/gdrive
   rclone config      # configure "gdrive" remote once
   rclone mount gdrive: ~/gdrive --vfs-cache-mode writes --daemon
   ```
5. **Add Kaggle credentials:**
   ```bash
   mkdir -p ~/.kaggle
   # upload kaggle.json via VS Code explorer into ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
6. **Download datasets:**
   ```bash
   python download_datasets.py
   ```
7. **Preprocess datasets:**
   ```bash
   python preprocess_datasets.py
   ```
8. All raw + processed files live in `~/gdrive/DepressionDetectionData/`.

## Configuration
- Toggle datasets and sampling limits in `config.py`.
- Processed outputs are saved as `.parquet` or `.npz` for easy ML ingestion.
```

---

## ☁️ Step 3: Push to GitHub
```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/depression-detection-data.git
git add .
git commit -m "Initial commit: Codespaces data pipeline"
git push -u origin main
```

---

## 💻 Step 4: Launch Codespace
1. On GitHub repo → **Code** button → **Codespaces** tab → **Create codespace**.
2. Wait for the environment to build (the devcontainer handles dependencies automatically).

---

## 🔐 Step 5: Configure Access Inside Codespace

### Kaggle
```bash
mkdir -p ~/.kaggle
# Upload kaggle.json via VS Code explorer into ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Google Drive (`rclone`)
```bash
rclone config     # create remote named "gdrive" (follow prompts)
mkdir -p ~/gdrive
rclone mount gdrive: ~/gdrive --vfs-cache-mode writes --daemon
ls ~/gdrive       # should list Drive contents
```

> Re-run the mount command whenever the Codespace restarts.

---

## 📥 Step 6: Download Datasets

```bash
python download_datasets.py
```

This pulls everything (RAVDESS, MELD, TESS, FER2013, etc.) directly into `~/gdrive/DepressionDetectionData/raw/`.

---

## 🧪 Step 7: Preprocess & Save

```bash
python preprocess_datasets.py
```

Outputs land in `~/gdrive/DepressionDetectionData/processed/`, e.g.:

- `ravdess_audio_features.parquet`
- `meld_text_features.parquet`
- `fer2013_images.npz`
- `tess_audio_features.parquet`
- `depression_reddit_text.parquet`
- `mental_health_survey.parquet`
- `mental_health_preprocessor.joblib`

---

## ✅ Step 8: Verify Results

```bash
ls ~/gdrive/DepressionDetectionData/processed/
du -sh ~/gdrive/DepressionDetectionData/*
```

You can open Parquet files directly in Codespaces for inspection or download them from Drive later.

---

## 🛎️ Tips & Best Practices
- Codespaces idle out; when you reconnect, just re-run the `rclone mount` command.
- Kaggle and Drive credentials stay inside the Codespace (not in git).
- Adjust toggles and sample limits in `config.py` to fit storage/time budgets.
- Use `git status` often to ensure large data files aren’t accidentally staged.

---

You now have a fully reproducible cloud setup: GitHub hosts your code, Codespaces processes the data, and Google Drive stores both raw and clean outputs—no local disk space sacrificed. Happy dataset wrangling!