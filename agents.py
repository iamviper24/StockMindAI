import os
from langchain_google_genai import ChatGoogleGenerativeAI
from models import GraphState, CorporateResearchOutput, FinancialAnalysisOutput, SentimentAnalysisOutput, InvestmentRecommendationOutput, ReportGenerationOutput
from services import fetch_yfinance_data, fetch_tavily_news, fetch_firecrawl_corporate
from analysis import calculate_technical_indicators, deduplicate_and_compress_news, process_corporate_documents
from config import logger

def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=api_key, temperature=0.0) 

# --- AGENT 1 ---
def market_intelligence_agent(state: GraphState) -> dict:
    logger.info(f"Agent 1: Fetching Market Data for {state['ticker']}")
    try:
        yf_data = fetch_yfinance_data(state['ticker'])
        hist_data = yf_data['history']
        
        ta_data = calculate_technical_indicators(hist_data)
        if "error" not in ta_data:
            tech_indicators = ta_data
            hist_data = ta_data.pop('df_with_ta')
        else:
            tech_indicators = {}
            
        return {
            "financial_metrics": yf_data['metrics'],
            "historical_data": hist_data,
            "intraday_data": yf_data.get('intraday', "[]"),
            "technical_indicators": tech_indicators,
            "news_data": yf_data['news']
        }
    except Exception as e:
        return {"error": f"Market Intel Error: {str(e)}"}

# --- AGENT 2 ---
def news_retrieval_agent(state: GraphState) -> dict:
    if state.get('error'): return {}
    logger.info(f"Agent 2: Fetching Market News for {state['ticker']}")
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    tavily_news = fetch_tavily_news(state['ticker'], tavily_key)
    
    all_news = state.get('news_data', []) + tavily_news
    
    return {"news_data": deduplicate_and_compress_news(all_news)}

# --- AGENT 3 ---
def corporate_research_agent(state: GraphState) -> dict:
    if state.get('error'): return {}
    logger.info(f"Agent 3: Corporate Research for {state['ticker']}")
    
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    raw_corp_data = fetch_firecrawl_corporate(state['ticker'], firecrawl_key)
    
    if not raw_corp_data:
        return {"corporate_research": None}
        
    processed_docs = process_corporate_documents(raw_corp_data)
    
    llm = get_llm().with_structured_output(CorporateResearchOutput)
    prompt = f"""
    Analyze the following official corporate documents and filings for {state['ticker']}.
    Extract critical business strategy, capital allocation, future guidance, and management risks strictly from the text.
    Do not invent or infer information. If missing, state 'Unavailable'.
    Documents: {processed_docs}
    """
    try:
        result = llm.invoke(prompt)
        return {"corporate_research": result}
    except Exception as e:
        logger.warning(f"Corporate Research extraction failed, bypassing: {str(e)}")
        return {"corporate_research": None}

# --- AGENT 4 ---
def financial_analysis_agent(state: GraphState) -> dict:
    if state.get('error'): return {}
    logger.info(f"Agent 4: Financial Analysis for {state['ticker']}")
    
    llm = get_llm().with_structured_output(FinancialAnalysisOutput)
    metrics = state['financial_metrics']
    corp_context = state.get('corporate_research')
    
    prompt = f"""
    Analyze the financial health of {state['ticker']}.
    Metrics: {metrics}
    Official Corporate Insights: {corp_context.dict() if corp_context else 'None available'}
    Provide an objective, institutional-grade financial health assessment. Use measurable facts. Do not use hyperbolic marketing language.
    """
    try:
        return {"financial_analysis": llm.invoke(prompt)}
    except Exception as e:
        return {"error": f"Financial Analysis Error: {str(e)}"}

# --- AGENT 5 ---
def market_sentiment_agent(state: GraphState) -> dict:
    if state.get('error'): return {}
    logger.info(f"Agent 5: Sentiment Analysis for {state['ticker']}")
    
    llm = get_llm().with_structured_output(SentimentAnalysisOutput)
    news = state['news_data']
    corp_context = state.get('corporate_research')
    
    prompt = f"""
    Analyze market sentiment for {state['ticker']}.
    News: {news}
    Official Press Releases & Filings: {corp_context.press_release_summary if corp_context else 'None'}
    Determine overall sentiment, key drivers, and market perception using objective tone.
    """
    try:
        return {"sentiment_analysis": llm.invoke(prompt)}
    except Exception as e:
        return {"error": f"Sentiment Analysis Error: {str(e)}"}

# --- AGENT 6 ---
def investment_advisor_agent(state: GraphState) -> dict:
    if state.get('error'): return {}
    logger.info(f"Agent 6: Investment Advisor for {state['ticker']}")
    
    llm = get_llm().with_structured_output(InvestmentRecommendationOutput)
    context = {
        "Ticker": state['ticker'], "Horizon": state['horizon'],
        "Financials": state['financial_analysis'].dict() if state.get('financial_analysis') else {},
        "Sentiment": state['sentiment_analysis'].dict() if state.get('sentiment_analysis') else {},
        "Technicals": state['technical_indicators'],
        "Corporate Strategy": state['corporate_research'].dict() if state.get('corporate_research') else {}
    }
    
    prompt = f"""Act as a tier-one institutional equity research analyst (e.g., Bloomberg, Goldman Sachs). 
    Based purely on the retrieved context below, provide a definitive investment recommendation for {state['ticker']} with a {state['horizon']} horizon. 
    Context: {context}. 
    Be strictly objective, factual, and concise. Explain factors with objective weights summing to 100."""
    try:
        return {"investment_recommendation": llm.invoke(prompt)}
    except Exception as e:
        return {"error": f"Investment Advisor Error: {str(e)}"}

# --- AGENT 7 ---
def report_generator_agent(state: GraphState) -> dict:
    if state.get('error'): return {}
    logger.info(f"Agent 7: Report Generation for {state['ticker']}")
    
    llm = get_llm().with_structured_output(ReportGenerationOutput)
    corp_data = state.get('corporate_research')
    metrics = state.get('financial_metrics', {})
    
    prompt = f"""
    You are an expert financial research analyst generating an institutional-quality equity research report for {state['ticker']}.

    GENERAL GUIDELINES:
    - High-information-density, concise, evidence-backed, and professional.
    - Every statement MUST be supported by the retrieved data provided below.
    - NEVER hallucinate, invent management commentary, or use general historical knowledge not retrieved here.
    - If information is unavailable from the data below, explicitly state: "Data unavailable from retrieved sources."
    - Tone must be objective. NO exaggerated adjectives (e.g., "massive", "extraordinary", "staggering"). Use measurable facts.
    - Do NOT repeat the same insights across multiple sections. Each section must introduce new information.
    - Prefer bullet points over lengthy narrative paragraphs.

    DATA CONTEXT:
    Raw Metrics: {metrics}
    Financials: {state['financial_analysis']}
    Sentiment: {state['sentiment_analysis']}
    Recommendation: {state['investment_recommendation']}
    Corporate Official Info: {corp_data.dict() if corp_data else 'Unavailable'}
    Technicals: {state['technical_indicators']}

    ENFORCE THIS EXACT MARKDOWN STRUCTURE:

    ## Executive Summary
    (Max 8–10 sentences covering overall financial health, technical outlook, news sentiment, investment recommendation, major risks, and confidence. Do not repeat detailed metrics here.)

    ## Metrics Snapshot
    (Format as a compact markdown table containing exactly these metrics derived from the raw data: Price, Market Cap, P/E, EPS, Dividend Yield. Add N/A if missing.)

    ## Company Overview
    (Max 2 short paragraphs on business model, core segments, competitive position. No generic statements.)

    ## Management Discussion
    (Summarize retrieved management priorities, strategic initiatives, capital allocation, guidance, key concerns. Do not fabricate.)

    ## Quarterly & Annual Highlights
    (Bullet points. Include Revenue, Net Income, EPS, Margins, Growth, Cash Flow, Guidance.)

    ## Business Strategy
    (Bullet points. E.g., Current Strategic Priorities like AI Integration, Expansion, etc.)

    ## Competitive Position
    (Focus on economic moat, market share, peer comparison, key differentiators. No generic praise.)

    ## Capital Allocation
    (Discuss Share Buybacks, Dividends, Cash Position, Debt, CapEx, R&D, Acquisitions.)


    ## Technical Analysis
    (Keep concise. Organize as Trend, Momentum, Support/Resistance, Moving Averages, MACD, RSI, ADX, Volume. End with 'Technical Outlook'.)

    ## News Summary
    (Bullet points for Positive Drivers, Negative Drivers, Market Sentiment, Analyst Consensus, Key Catalysts.)

    ## Risk Assessment
    (Split into categories: Company Risks, Financial Risks, Technical Risks, Macro Risks, Regulatory Risks. Bullet points for each.)

    ## Investment Recommendation
    (Use this EXACT format)
    **Recommendation:** BUY / HOLD / SELL
    **Confidence:** XX%
    **Risk Score:** XX%
    **Investment Horizon:** {state['horizon']}
    
    **Bull Case**
    * ...
    **Bear Case**
    * ...
    **Key Catalysts**
    * ...
    **Watch List**
    * ...
    """
    try:
        res = llm.invoke(prompt)
        return {"generated_report_md": res.markdown_report}
    except Exception as e:
        return {"error": f"Report Generation Error: {str(e)}"}