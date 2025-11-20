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