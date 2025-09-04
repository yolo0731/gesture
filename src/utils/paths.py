from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk up to locate the project root (containing run.py or README.md)."""
    for parent in [start] + list(start.parents):
        if (parent / 'run.py').exists() or (parent / 'README.md').exists():
            return parent
    # Fallback for src-layout: package/.../utils -> project is three levels up
    return start.parents[3]


PACKAGE_DIR = Path(__file__).resolve().parent  # src/utils
SRC_DIR = PACKAGE_DIR.parent                   # src
PROJECT_ROOT = _find_project_root(SRC_DIR)


def models_dir() -> Path:
    return PROJECT_ROOT / 'models'


def data_dir() -> Path:
    return PROJECT_ROOT / 'data'


def ui_dir() -> Path:
    # UI .ui resources live under src/ui/resources
    return SRC_DIR / 'ui' / 'resources'
