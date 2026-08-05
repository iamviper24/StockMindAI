import streamlit as st
import os
from config import REPORTS_DIR
from utils import WatchlistManager, save_json
from services import get_market_movers, get_watchlist_data, get_company_name
from graph import run_analysis_workflow
from rag import ingest_report_to_faiss, chat_with_report
from analysis import generate_master_chart, generate_sparkline
import plotly.graph_objects as go

st.set_page_config(page_title="AI Global Stock Research", layout="wide", page_icon="📈")

def safe_render(text: str) -> str:
    if not text: return ""
    return str(text).replace('$', r'\$')

def format_market_cap(val: float, sym: str) -> str:
    if not val or val == 0: return "-"
    if val >= 1e12: return f"{sym}{val/1e12:.2f}T"
    if val >= 1e9: return f"{sym}{val/1e9:.2f}B"
    if val >= 1e6: return f"{sym}{val/1e6:.2f}M"
    return f"{sym}{val:,.2f}"

def render_horizontal_card(data: dict, prefix: str, is_watchlist: bool = False):
    if not data: return
    is_up = data['change'] >= 0
    color = "#00C805" if is_up else "#FF333A"
    sign = "+" if is_up else ""
    curr = data.get('currency', '$')
    
    display_name = data['name']
    if len(display_name) > 22: display_name = display_name[:20] + "..."
    
    with st.container(border=True):
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown(
                f"**{display_name}**<br>"
                f"<span style='font-size:12px; color:gray;'>{data['ticker']}</span><br>"
                f"<b>{curr}{data['price']:.2f}</b><br>"
                f"<span style='font-size:13px; color:{color};'>{sign}{data['change']:.2f} ({sign}{data['change_pct']:.2f}%)</span>", 
                unsafe_allow_html=True
            )
        with c2:
            fig = generate_sparkline(data['prices'], is_up)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True}, key=f"spark_{prefix}_{data['ticker']}")
            
        if is_watchlist:
            if st.button("❌ Remove", key=f"del_{data['ticker']}", use_container_width=True):
                WatchlistManager.remove_ticker(data['ticker'])
                st.rerun()

def get_recommendation_color(rec: str) -> str:
    r = rec.upper()
    if "STRONG BUY" in r: return "#00FF00"
    if "BUY" in r: return "#00C805"
    if "STRONG SELL" in r: return "#FF0000"
    if "SELL" in r: return "#FF333A"
    return "#FFA500"

#Initialization
if "state" not in st.session_state: st.session_state.state = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

gemini_key = os.getenv("GEMINI_API_KEY", "")

#Live market dashboard
st.title(" AI-Powered Global Stock Research")

selected_region = st.radio("Select Market Region", ["India", "USA", "Europe", "Japan", "China/HK"], horizontal=True)

with st.expander(f"Market Movers Overview ({selected_region})", expanded=True):
    movers_data = get_market_movers(selected_region)
    
    st.markdown(f"#####  Top Gainers Today")
    cols_g = st.columns(4)
    for i, g in enumerate(movers_data.get('gainers', [])[:4]):
        with cols_g[i]: render_horizontal_card(g, "gainer")
        
    st.markdown(f"#####  Top Losers Today")
    cols_l = st.columns(4)
    for i, l in enumerate(movers_data.get('losers', [])[:4]):
        with cols_l[i]: render_horizontal_card(l, "loser")

    st.markdown(f"#####  Most Active (Volume)")
    cols_a = st.columns(4)
    for i, a in enumerate(movers_data.get('active', [])[:4]):
        with cols_a[i]: render_horizontal_card(a, "active")

st.divider()

#Watchlist
st.markdown("###  My Watchlist")

wl_objects = WatchlistManager.get_watchlist()
wl_symbols = [obj["ticker"] for obj in wl_objects]
wl_data = get_watchlist_data(wl_symbols)

for i in range(0, len(wl_data), 4):
    chunk = wl_data[i:i+4]
    cols = st.columns(4)
    for j, data in enumerate(chunk):
        with cols[j]:
            render_horizontal_card(data, "watch", is_watchlist=True)

st.write("") 
c_add1, c_add2, _ = st.columns([1, 1, 2])
with c_add1:
    new_ticker = st.text_input("Add to Watchlist", placeholder="e.g. AAPL or ITC.NS", label_visibility="collapsed")
with c_add2:
    if st.button("Add Ticker", use_container_width=True):
        if new_ticker:
            tick = new_ticker.upper()
            with st.spinner("Resolving company name..."):
                c_name = get_company_name(tick)
                WatchlistManager.add_ticker(tick, c_name)
            st.rerun()

st.divider()

#Report Generation
st.subheader("🤖 Generate AI Analysis Report")

c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
ticker_input = c1.text_input("Stock Symbol", value="RELIANCE.NS", key="report_ticker").upper()
analysis_type = c2.selectbox("Analysis Type", ["Comprehensive", "Technical", "Fundamental"])
horizon = c3.selectbox("Investment Horizon", [
    "Intraday (1 Day)", 
    "Swing (1 Week)", 
    "Short-Term (1-3 Mo)", 
    "Medium-Term (6-12 Mo)", 
    "Long-Term (1-5 Yr)"
])

with c4:
    st.write("")
    st.write("") 
    analyze_btn = st.button("Analyze Stock", type="primary", use_container_width=True)

if analyze_btn:
    if not gemini_key:
        st.error("Please add your GEMINI_API_KEY to the .env file to run analyses.")
    else:
        with st.spinner(f"Running multi-agent AI analysis for {ticker_input}..."):
            result_state = run_analysis_workflow(ticker_input, analysis_type, horizon)
            
            if result_state.get("error"):
                st.error(result_state["error"])
            else:
                md_path = REPORTS_DIR / f"{ticker_input}.md"
                json_path = REPORTS_DIR / f"{ticker_input}.json"
                
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(result_state["generated_report_md"])
                    

                clean_state = {k: v for k, v in result_state.items() if k not in ["historical_data", "intraday_data", "short_term"]}
                
                for key in ['corporate_research', 'financial_analysis', 'sentiment_analysis', 'investment_recommendation']:
                    if clean_state.get(key):
                        try:
                            clean_state[key] = clean_state[key].model_dump()
                        except AttributeError:
                            clean_state[key] = clean_state[key].dict() 
                
                save_json(json_path, clean_state)
                ingest_report_to_faiss(ticker_input, result_state["generated_report_md"])
                
                st.session_state.state = result_state
                st.session_state.chat_history = []
                st.success(f"Analysis Complete for {ticker_input}!")

#Report View
if st.session_state.state and not st.session_state.state.get("error"):
    s = st.session_state.state
    fm = s["financial_metrics"]
    adv = s["investment_recommendation"]
    
    curr = fm.get('currency_symbol', '$')
    
    st.divider()
    
    report_company_name = fm.get('company_name', s['ticker'])
    st.markdown(f"<h2 style='margin-bottom: 0px;'>Report: {report_company_name}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color:gray; margin-top: 5px; margin-bottom: 20px;'>{s['ticker']}</h5>", unsafe_allow_html=True)
    
    price = fm.get('price', 0)
    change = fm.get('change_pct', 0)
    pe = fm.get('pe_ratio', 0)
    eps = fm.get('eps', 0)
    div = fm.get('dividend_yield', 0)
    
    price_str = f"{curr}{price:.2f}" if price else "-"
    change_str = f"{change:.2f}%" if price else None
    pe_str = f"{pe:.2f}" if pe else "-"
    eps_str = f"{curr}{eps:.2f}" if eps else "-"
    div_str = f"{div:.2f}%" if div else "-"
    
    cols = st.columns(6)
    cols[0].metric("Current Price", price_str, change_str)
    cols[1].metric("Market Cap", format_market_cap(fm.get('market_cap', 0), curr))
    cols[2].metric("PE Ratio", pe_str)
    cols[3].metric("EPS", eps_str)
    cols[4].metric("Div Yield", div_str)
    
    rec_color = get_recommendation_color(adv.recommendation)
    with cols[5]:
        st.markdown(
            f"""
            <div style="line-height: 1.2; margin-top: -5px;">
                <span style="font-size: 14px; color: rgb(166, 170, 184); font-weight: 400;">Recommendation</span><br>
                <span style="font-size: 2.2rem; font-weight: 600; color: {rec_color};">{adv.recommendation}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    st.divider()

    st.subheader(f"Interactive Advanced Chart")
    
    chart_ui_1, chart_ui_2 = st.columns([1, 1])
    with chart_ui_1:
        default_tf_index = 0
        h_text = s['horizon']
        if "1 Week" in h_text: default_tf_index = 1
        elif "1-3 Mo" in h_text: default_tf_index = 3
        elif "6-12 Mo" in h_text: default_tf_index = 5
        elif "1-5 Yr" in h_text: default_tf_index = 6
        
        selected_tf = st.radio("Timeframe", ["1D", "1W", "1M", "3M", "6M", "1Y", "5Y"], index=default_tf_index, horizontal=True, label_visibility="collapsed")
    with chart_ui_2:
        selected_style = st.radio("Chart Type", ["Mountain", "Line", "Candle", "Bar"], index=0, horizontal=True, label_visibility="collapsed")
    
    short_term_data = s.get('short_term', "[]")
    prev_c = fm.get('prev_close', 0)
    
    main_chart = generate_master_chart(s['historical_data'], s['intraday_data'], short_term_data, selected_tf, selected_style, prev_c)
    st.plotly_chart(main_chart, use_container_width=True)

    st.divider()
    
    st.markdown(f"### AI Investment Advisor: <span style='color:{rec_color}'>{adv.recommendation}</span>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence Score", f"{adv.overall_confidence}/100")
    c2.metric("Risk Score", f"{adv.overall_risk}/100")
    c3.metric("Target Horizon", adv.target_horizon)
    
    st.markdown(f"**Bull Case:** {safe_render(adv.bull_case)}")
    st.markdown(f"**Bear Case:** {safe_render(adv.bear_case)}")
    
    st.markdown("**Key Reasons for Recommendation:**")
    for reason in adv.key_reasons:
        st.markdown(f"- {safe_render(reason)}")
        
    st.divider()
    
    tab1, tab2, tab3 = st.tabs([
        "📄 AI Report", "💼 Financials & Risk", "📰 News & Sentiment"
    ])
    
    with tab1:
        st.markdown(safe_render(s["generated_report_md"]))
        
    with tab2:
        fin = s["financial_analysis"]
        st.subheader("Financial Health")
        st.progress(fin.financial_confidence / 100, text=f"Financial Confidence: {fin.financial_confidence}%")
        st.progress(fin.financial_risk / 100, text=f"Financial Risk: {fin.financial_risk}%")
        
        st.markdown(safe_render(fin.summary))
        st.markdown(f"**QoQ Comparison:** {safe_render(fin.quarterly_comparison)}")
        st.markdown(f"**Revenue:** {safe_render(fin.revenue_analysis)}")
        st.markdown(f"**Profitability:** {safe_render(fin.profitability_analysis)}")
        st.markdown(f"**Debt & Cash:** {safe_render(fin.debt_and_cash)}")
        
    with tab3:
        sent = s["sentiment_analysis"]
        st.subheader(f"Overall Sentiment: {sent.overall_sentiment}")
        st.progress(sent.news_confidence / 100, text=f"Sentiment Confidence: {sent.news_confidence}%")
        st.markdown(safe_render(sent.news_summary))
        st.markdown(f"**Market Perception:** {safe_render(sent.market_perception)}")
        
        st.subheader("Recent News")
        valid_news_found = False
        for n in s.get("news_data", []):
            title = n.get('title')
            link = n.get('link', '#')
            publisher = n.get('publisher', '')
            publisher_display = f" ({publisher})" if publisher and publisher != "Tavily" else ""
            
            if title and str(title).strip() != "None":
                valid_news_found = True
                st.markdown(f"- [{safe_render(title)}]({link}){publisher_display}")
                
        if not valid_news_found:
            st.info("No recent valid news articles found for this stock.")
    
    st.divider()
    st.subheader("💬 Ask the AI Assistant")
    st.caption("Answers are generated using the local analysis report and real-time live web data.")
    
    chat_container = st.container(height=400, border=True)
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Say hello! You can ask questions like 'Who are their main competitors?' or 'What are the main risks?'")
            
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(safe_render(msg["content"]))
                
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([8, 1])
        with col_input:
            user_query = st.text_input("Ask a question...", label_visibility="collapsed", placeholder="Ask a question about this stock...")
        with col_btn:
            submit_chat = st.form_submit_button("Send", use_container_width=True)

    if submit_chat and user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.spinner("Analyzing..."):
            ans = chat_with_report(s["ticker"], user_query)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
        st.rerun()