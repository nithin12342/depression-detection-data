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