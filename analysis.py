import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from io import StringIO

@st.cache_resource
def get_cached_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_corporate_documents(docs: list) -> list:
    if not docs: return []
    for d in docs:
        title = d.get('title', '').lower()
        score = 50
        if 'annual' in title or '10-k' in title: score += 40
        elif 'quarterly' in title or '10-q' in title or 'earnings' in title: score += 30
        elif 'presentation' in title or 'investor' in title: score += 20
        elif 'press release' in title: score += 10
        d['authority_score'] = score
        
    docs = sorted(docs, key=lambda x: x['authority_score'], reverse=True)
    embeddings = get_cached_embeddings()
    texts = [d.get("summary", "")[:1000] for d in docs]
    try:
        vecs = embeddings.embed_documents(texts)
        keep = []
        for i, v1 in enumerate(vecs):
            is_dup = False
            for j in keep:
                v2 = vecs[j]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if sim > 0.85:
                    is_dup = True
                    break
            if not is_dup: keep.append(i)
        docs = [docs[i] for i in keep]
    except Exception: pass 
    
    for d in docs:
        content = d.get('summary', '')
        if len(content) > 3000:
            d['summary'] = content[:3000] + "\n[Content Compressed]"
    return docs[:5]

def calculate_technical_indicators(df_json: str) -> dict:
    df = pd.read_json(StringIO(df_json))
    if len(df) < 200: return {"error": "Not enough data for technical analysis"}

    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    recent = df.tail(60)
    support = recent['Low'].min()
    resistance = recent['High'].max()
    latest = df.iloc[-1]
    
    return {
        "sma_50": float(latest['SMA_50']), "sma_200": float(latest['SMA_200']),
        "rsi": float(latest['RSI']), "macd": float(latest['MACD']),
        "adx": float(latest['ADX']), "atr": float(latest['ATR']),
        "obv": float(latest['OBV']), "bb_high": float(latest['BB_High']),
        "bb_low": float(latest['BB_Low']), "support": float(support),
        "resistance": float(resistance), "df_with_ta": df.to_json(orient="records")
    }

def generate_master_chart(df_hist_json: str, df_intra_json: str, df_short_json: str, timeframe: str, chart_type: str, prev_close: float) -> go.Figure:
    is_intraday_view = False
    
    if timeframe == "1D":
        is_intraday_view = True
        df = pd.read_json(StringIO(df_intra_json))
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Datetime'])
            
    elif timeframe in ["1W", "1M"]:
        df = pd.read_json(StringIO(df_short_json))
        
        if df.empty:
            df = pd.read_json(StringIO(df_hist_json))
            if not df.empty: df['Date'] = pd.to_datetime(df['Date'])
        else:
            is_intraday_view = True 
            df['Date'] = pd.to_datetime(df['Datetime'])
            
        if not df.empty:
            end_date = df['Date'].max()
            delta = pd.DateOffset(weeks=1) if timeframe == "1W" else pd.DateOffset(months=1)
            df = df[df['Date'] >= (end_date - delta)]
            
    else:
        df = pd.read_json(StringIO(df_hist_json))
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            end_date = df['Date'].max()
            if timeframe == "3M": df = df[df['Date'] >= (end_date - pd.DateOffset(months=3))]
            elif timeframe == "6M": df = df[df['Date'] >= (end_date - pd.DateOffset(months=6))]
            elif timeframe == "1Y": df = df[df['Date'] >= (end_date - pd.DateOffset(years=1))]
            # 5Y requires no slicing

    # If completely empty, return empty chart
    if df.empty: return go.Figure()

    #Chart Builder
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2])
    
    current_price = df['Close'].iloc[-1]
    compare_price = prev_close if timeframe == "1D" else df['Open'].iloc[0]
    is_up = current_price >= compare_price
    
    color = "#00C805" if is_up else "#FF333A"
    fill_color = "rgba(0, 200, 5, 0.1)" if is_up else "rgba(255, 51, 58, 0.1)"

    
    if chart_type in ["Candle", "Bar"]:
        y_min, y_max = df['Low'].min(), df['High'].max()
    else:
        y_min, y_max = df['Close'].min(), df['Close'].max()
        
    if timeframe == "1D" and prev_close > 0:
        y_min = min(y_min, prev_close)
        y_max = max(y_max, prev_close)
        
    padding = (y_max - y_min) * 0.1 if y_max != y_min else 5
    base_y = y_min - padding
    
    if chart_type == "Candle":
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    elif chart_type == "Bar":
        fig.add_trace(go.Ohlc(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', line=dict(color=color, width=2), name='Price'), row=1, col=1)
    else: 
        # Mountain view strictly bound to base_y so it stretches elegantly
        fig.add_trace(go.Scatter(x=df['Date'], y=[base_y]*len(df), mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'), hoverinfo='skip', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', line=dict(color=color, width=2), fill='tonexty', fillcolor=fill_color, name='Price'), row=1, col=1)

    if timeframe == "1D" and prev_close > 0:
        fig.add_hline(y=prev_close, line_dash="dash", line_color="gray", opacity=0.6, row=1, col=1, annotation_text="Prev Close", annotation_position="bottom left")

    vol_colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=vol_colors, name='Volume'), row=2, col=1)

    # Dynamic Formatting
    x_format = "%b %d, %H:%M" if is_intraday_view else "%b %d, %Y"
    if timeframe == "1D": x_format = "%I:%M %p"
    
    fig.update_layout(
        margin=dict(l=0, r=40, t=10, b=10), height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(side='right', showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[base_y, y_max + padding]), 
        xaxis2=dict(showgrid=False, tickformat=x_format, color="white"), yaxis2=dict(showgrid=False, showticklabels=False),
        hovermode="x unified"
    )
    return fig

def deduplicate_and_compress_news(news_list: list) -> list:
    if not news_list: return []
    texts = [f"{n.get('title', '')} {n.get('summary', '')}" for n in news_list]
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    except:
        return news_list[:10]

    keep_indices = []
    for i in range(len(texts)):
        is_duplicate = False
        for j in keep_indices:
            if cosine_sim[i][j] > 0.7:
                is_duplicate = True
                break
        if not is_duplicate: keep_indices.append(i)

    ranked_news = [news_list[i] for i in keep_indices]
    for n in ranked_news:
        summary = n.get('summary', '')
        if summary and len(summary) > 500: n['summary'] = summary[:500] + "..."
            
    return ranked_news[:10]

def generate_sparkline(prices: list, is_up: bool) -> go.Figure:
    if not prices: return go.Figure()
    color = "#00C805" if is_up else "#FF333A"
    fill_color = "rgba(0, 200, 5, 0.15)" if is_up else "rgba(255, 51, 58, 0.15)"
    min_p, max_p = min(prices), max(prices)
    padding = (max_p - min_p) * 0.1 if max_p != min_p else 1
    
    fig = go.Figure()
    x_vals = list(range(len(prices)))
    
    fig.add_trace(go.Scatter(x=x_vals, y=[min_p - padding]*len(prices), mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_vals, y=prices, mode='lines', line=dict(color=color, width=2), fill='tonexty', fillcolor=fill_color, hoverinfo='skip'))
    
    max_x = len(prices) + max(1, int(len(prices) * 0.05))
    
    fig.update_layout(
        margin=dict(l=0, r=20, t=5, b=5), height=60, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, 
        xaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False, range=[0, max_x]), 
        yaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False, range=[min_p - padding, max_p + padding])
    )
    return fig