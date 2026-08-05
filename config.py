import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
FAISS_DB_DIR = BASE_DIR / "faiss_db"
WATCHLIST_FILE = BASE_DIR / "watchlist.json"

# Ensure directories exist
for directory in [CACHE_DIR, REPORTS_DIR, FAISS_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StockAI")

def get_env_var(var_name: str) -> str:
    return os.getenv(var_name, "")
