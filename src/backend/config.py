from pathlib import Path

# Project root: .../TFG
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Shared data directory for backend persistence
DATA_DIR = BASE_DIR / "data"
# SQLite database used by backend DAO
APP_DB_PATH = DATA_DIR / "app_database.sqlite"
# FTS5 autocomplete index (separate from APP_DB_PATH); fixed path per project plan.
AUTOCOMPLETE_FTS_DB_PATH = Path(DATA_DIR) / "product_autocomplete_fts.sqlite"