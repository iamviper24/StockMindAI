import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import WATCHLIST_FILE

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    elif pd.isna(obj) if isinstance(obj, float) else False:
        return None
    return obj

def save_json(filepath: Path, data: dict):
    clean_data = sanitize_for_json(data)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=4)

def load_json(filepath: Path) -> dict:
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

class WatchlistManager:
    @staticmethod
    def get_watchlist() -> list:
        data = load_json(WATCHLIST_FILE)
        tickers = data.get("tickers", [])
        
        if not tickers:
            tickers = [
                {"ticker": "RELIANCE.NS", "name": "Reliance Industries Limited"},
                {"ticker": "TCS.NS", "name": "Tata Consultancy Services Limited"},
                {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Limited"},
                {"ticker": "INFY.NS", "name": "Infosys Limited"}
            ]
            data["tickers"] = tickers
            save_json(WATCHLIST_FILE, data)
            return tickers

        if tickers and isinstance(tickers[0], str):
            tickers = [{"ticker": t, "name": t} for t in tickers]
            data["tickers"] = tickers
            save_json(WATCHLIST_FILE, data)
            
        return tickers

    @staticmethod
    def add_ticker(ticker: str, name: str):
        data = load_json(WATCHLIST_FILE)
        tickers = data.get("tickers", [])
        # Prevent duplicates
        if not any(t.get("ticker") == ticker for t in tickers):
            tickers.append({"ticker": ticker, "name": name})
            data["tickers"] = tickers
            save_json(WATCHLIST_FILE, data)

    @staticmethod
    def remove_ticker(ticker: str):
        data = load_json(WATCHLIST_FILE)
        tickers = data.get("tickers", [])
        tickers = [t for t in tickers if t.get("ticker") != ticker]
        data["tickers"] = tickers
        save_json(WATCHLIST_FILE, data)