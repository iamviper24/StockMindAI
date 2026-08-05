import yfinance as yf
import requests
import json
import concurrent.futures
from typing import Dict, Any, List
import streamlit as st
import pandas as pd

def get_currency_symbol(ticker: str) -> str:
    """Infers the currency symbol based on the stock exchange suffix."""
    t = ticker.upper()
    if t.endswith('.NS') or t.endswith('.BO'): return '₹'
    if t.endswith('.L'): return '£'
    if t.endswith('.DE') or t.endswith('.PA') or t.endswith('.AS') or t.endswith('.MI') or t.endswith('.MC'): return '€'
    if t.endswith('.SW'): return 'CHF '
    if t.endswith('.T'): return '¥'
    if t.endswith('.HK'): return 'HK$'
    if t.endswith('.TO') or t.endswith('.V') or t.endswith('.CN'): return 'C$'
    if t.endswith('.AX'): return 'A$'
    return '$' 

@st.cache_data(ttl=86400)
def get_company_name(ticker: str) -> str:
    indices = {"^NSEI": "NIFTY 50", "^BSESN": "SENSEX", "^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^STOXX50E": "EURO STOXX 50", "^N225": "NIKKEI 225", "^HSI": "HANG SENG"}
    if ticker in indices: return indices[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName")
        return name if name else ticker
    except: return ticker

@st.cache_data(ttl=900)
def fetch_yfinance_data(ticker: str) -> Dict[str, Any]:
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Standard 5-Year Daily data
    hist = stock.history(period="5y")
    if hist.empty: raise ValueError(f"No historical data found for {ticker}")
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    
    # 1-Day Intraday data (5-minute intervals)
    intraday = stock.history(period="1d", interval="5m")
    intraday_json = "[]"
    if not intraday.empty:
        intraday.reset_index(inplace=True)
        # Normalize column name to Datetime to prevent silent crashes
        if 'Date' in intraday.columns and 'Datetime' not in intraday.columns:
            intraday.rename(columns={'Date': 'Datetime'}, inplace=True)
        if 'Datetime' in intraday.columns:
            intraday['Datetime'] = intraday['Datetime'].astype(str)
        intraday_json = intraday.to_json(orient="records")
        
    # High-resolution Short-Term data for 1W/1M (1-hour intervals)
    short_term = stock.history(period="1mo", interval="1h")
    short_term_json = "[]"
    if not short_term.empty:
        short_term.reset_index(inplace=True)
        # Normalize column name to Datetime for non-US markets
        if 'Date' in short_term.columns and 'Datetime' not in short_term.columns:
            short_term.rename(columns={'Date': 'Datetime'}, inplace=True)
        if 'Datetime' in short_term.columns:
            short_term['Datetime'] = short_term['Datetime'].astype(str)
        short_term_json = short_term.to_json(orient="records")
    
    try: bs, cf, qr = stock.balance_sheet.to_dict(), stock.cashflow.to_dict(), stock.quarterly_financials.to_dict()
    except Exception: bs, cf, qr = {}, {}, {}

    curr_code = info.get("currency", "USD").upper()
    curr_map = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CAD": "C$", "AUD": "A$", "HKD": "HK$", "CHF": "CHF "}
    
    metrics = {
        "company_name": info.get("shortName") or info.get("longName") or ticker,
        "currency_symbol": curr_map.get(curr_code, f"{curr_code} "),
        "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "prev_close": info.get("previousClose", 0),
        "change_pct": info.get("regularMarketChangePercent", 0) * 100,
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", 0),
        "eps": info.get("trailingEps", 0),
        "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
        "sector": info.get("sector", "Unknown"), "industry": info.get("industry", "Unknown"),
        "balance_sheet": bs, "cash_flow": cf, "quarterly_results": qr,
        "institutional_holders": info.get("institutionsCount", 0)
    }
    
    news = []
    for n in stock.news:
        title = n.get("title")
        link = n.get("link")
        if title and link and str(title).strip() != "None":
            news.append({"title": title, "publisher": n.get("publisher", "Yahoo Finance"), "link": link})
            
    return {
        "metrics": metrics, 
        "history": hist.to_json(orient="records"), 
        "intraday": intraday_json, 
        "short_term": short_term_json,
        "news": news[:5]
    }

@st.cache_data(ttl=1800)
def fetch_tavily_news(ticker: str, api_key: str) -> List[Dict[str, str]]:
    if not api_key: return []
    try:
        res = requests.post("https://api.tavily.com/search", json={"api_key": api_key, "query": f"{ticker} stock market news analyst opinions", "search_depth": "advanced"}, timeout=10)
        res.raise_for_status()
        return [{"title": r.get("title"), "publisher": "Tavily", "link": r.get("url"), "summary": r.get("content")} for r in res.json().get("results", [])]
    except Exception: return []

@st.cache_data(ttl=86400)
def fetch_firecrawl_corporate(ticker: str, api_key: str) -> List[Dict[str, str]]:
    if not api_key: return []
    try:
        query = f'"{ticker}" investor relations OR annual report OR quarterly report OR press release OR exchange filings site:.com OR site:.in'
        res = requests.post("https://api.firecrawl.dev/v0/scrape", headers={"Authorization": f"Bearer {api_key}"}, json={"url": f"https://www.google.com/search?q={query}"}, timeout=10)
        res.raise_for_status()
        content = res.json().get("data", {}).get("markdown", "")
        return [{"title": f"{ticker} Official Corporate Filings", "type": "Corporate", "source": "Firecrawl", "url": "", "summary": content[:8000]}]
    except Exception: return []

def _process_market_batch(symbols: List[str]) -> List[Dict]:
    if not symbols: return []
    try:
        df = yf.download(symbols, period="1d", interval="15m", group_by="ticker", progress=False)
        data = []
        names_dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            future_to_sym = {executor.submit(get_company_name, sym): sym for sym in symbols}
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                try: names_dict[sym] = future.result()
                except: names_dict[sym] = sym
                
        for sym in symbols:
            try: sym_data = df[sym].dropna() if len(symbols) > 1 else df.dropna()
            except KeyError: continue
            if sym_data.empty: continue
            
            open_val = sym_data['Open'].iloc[0]
            close_val = sym_data['Close'].iloc[-1]
            vol = sym_data['Volume'].sum()
            prices = sym_data['Close'].tolist()
            change = close_val - open_val
            change_pct = (change / open_val) * 100 if open_val > 0 else 0
            
            display_name = sym.replace(".NS", "")
            if sym == "^NSEI": display_name = "NIFTY 50"
            elif sym == "^BSESN": display_name = "SENSEX"
            elif sym == "^GSPC": display_name = "S&P 500"
            elif sym == "^IXIC": display_name = "NASDAQ"
            
            data.append({
                "ticker": display_name, "original_ticker": sym, "name": names_dict.get(sym, display_name),
                "currency": get_currency_symbol(sym), "price": float(close_val),
                "change": float(change), "change_pct": float(change_pct),
                "volume": float(vol), "prices": [float(p) for p in prices]
            })
        return data
    except Exception: return []

@st.cache_data(ttl=300) 
def get_market_movers(region: str) -> Dict[str, List[Dict]]:
    headers = {'User-Agent': 'Mozilla/5.0'}
    dynamic_symbols = set()
    region_map = {
        "India": {"code": "IN", "core": ["^NSEI", "^BSESN", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "TITAN.NS", "ASIANPAINT.NS"]},
        "USA": {"code": "US", "core": ["^GSPC", "^IXIC", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "BRK-B", "LLY", "AVGO", "JPM", "XOM", "UNH", "V", "PG", "MA"]},
        "Europe": {"code": "GB", "core": ["^STOXX50E", "ASML.AS", "MC.PA", "SAP.DE", "AZN.L", "SHEL.L", "NOVN.SW", "HSBA.L", "TTE.PA", "OR.PA", "SAN.PA", "SIE.DE", "IBE.MC", "SU.PA", "AIR.PA", "ALV.DE", "BMW.DE", "VOW3.DE", "ENEL.MI", "PRX.AS", "UNA.AS", "BNP.PA"]},
        "Japan": {"code": "JP", "core": ["^N225", "7203.T", "6758.T", "8306.T", "9984.T", "9432.T", "6861.T", "8035.T", "4063.T", "6098.T", "7974.T", "8058.T", "6981.T", "4502.T", "7267.T", "6501.T", "8001.T", "6902.T", "8316.T", "4568.T", "9433.T"]},
        "China/HK": {"code": "HK", "core": ["^HSI", "0700.HK", "9988.HK", "3690.HK", "0941.HK", "0005.HK", "1299.HK", "0883.HK", "0386.HK", "1398.HK", "0939.HK", "3988.HK", "2318.HK", "1810.HK", "1211.HK", "0011.HK", "0388.HK", "0016.HK", "0857.HK", "0267.HK", "1928.HK", "0027.HK"]}
    }
    r_info = region_map.get(region, region_map["India"])

    try:
        res = requests.get(f"https://query1.finance.yahoo.com/v1/finance/trending/{r_info['code']}?count=15", headers=headers, timeout=5)
        if res.status_code == 200:
            for q in res.json().get('finance', {}).get('result', [])[0].get('quotes', []): dynamic_symbols.add(q['symbol'])
    except Exception: pass

    if region == "USA":
        for scr in ["day_gainers", "day_losers", "most_actives"]:
            try:
                url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&scrIds={scr}&count=10"
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code == 200:
                    for q in res.json().get('finance', {}).get('result', [])[0].get('quotes', []): dynamic_symbols.add(q['symbol'])
            except Exception: pass

    dynamic_symbols.update(r_info['core'])
    data = _process_market_batch(list(dynamic_symbols))
    
    stocks_only = [d for d in data if not d['original_ticker'].startswith('^')]
    gainers = sorted([d for d in stocks_only if d['change'] > 0], key=lambda x: x['change_pct'], reverse=True)[:4]
    losers = sorted([d for d in stocks_only if d['change'] < 0], key=lambda x: x['change_pct'])[:4]
    active = sorted(stocks_only, key=lambda x: x['volume'], reverse=True)[:4]
    return {"gainers": gainers, "losers": losers, "active": active}

@st.cache_data(ttl=60) 
def get_watchlist_data(watchlist_symbols: List[str]) -> List[Dict]:
    data = _process_market_batch(watchlist_symbols)
    ordered_data = []
    for w in watchlist_symbols:
        found = next((item for item in data if item["original_ticker"] == w), None)
        if found: ordered_data.append(found)
    return ordered_data