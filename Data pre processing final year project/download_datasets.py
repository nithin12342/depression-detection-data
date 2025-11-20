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