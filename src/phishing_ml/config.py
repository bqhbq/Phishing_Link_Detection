"""Cấu hình dùng chung cho 4 notebook trong `notebooks_new/`."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

RANDOM_STATE = 42
LABELS = ['Legitimate', 'Phishing']
PALETTE = {'Legitimate': '#2b8cbe', 'Phishing': '#e34a33'}


def find_project_root(start: Path | None = None) -> Path:
    """Đi ngược cây thư mục tới khi gặp data/raw/phishing_site_urls.csv."""
    start = Path.cwd() if start is None else Path(start)
    for p in [start, *start.parents]:
        if (p / 'data' / 'raw' / 'phishing_site_urls.csv').exists():
            return p
    raise FileNotFoundError('Không tìm thấy thư mục gốc dự án (data/raw/phishing_site_urls.csv).')


ROOT = find_project_root(Path(__file__).resolve().parent)
RAW_PATH = ROOT / 'data' / 'raw' / 'phishing_site_urls.csv'

# Toàn bộ object trung gian giữa các phase được lưu ở đây
ARTIFACT_DIR = ROOT / 'artifacts'
DATA_DIR = ARTIFACT_DIR / 'data'             # parquet / npz
TRANSFORMER_DIR = ARTIFACT_DIR / 'transformers'
MODEL_DIR = ARTIFACT_DIR / 'models'
REPORT_DIR = ARTIFACT_DIR / 'reports'        # csv / json kết quả


def ensure_dirs() -> None:
    for d in (DATA_DIR, TRANSFORMER_DIR, MODEL_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
