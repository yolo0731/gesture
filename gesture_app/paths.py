from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

def models_dir() -> Path:
    return PROJECT_ROOT / 'models'

def data_dir() -> Path:
    return PROJECT_ROOT / 'data'

def ui_dir() -> Path:
    return PACKAGE_DIR / 'ui'

