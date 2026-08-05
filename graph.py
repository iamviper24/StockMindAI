from langgraph.graph import StateGraph, END
from models import GraphState
from agents import (
    market_intelligence_agent,
    news_retrieval_agent,
    corporate_research_agent,
    financial_analysis_agent,
    market_sentiment_agent,
    investment_advisor_agent,
    report_generator_agent
)

def build_workflow():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("market_intel", market_intelligence_agent)
    workflow.add_node("news_retrieval", news_retrieval_agent)
    workflow.add_node("corporate_research", corporate_research_agent)
    workflow.add_node("financial_analysis", financial_analysis_agent)
    workflow.add_node("market_sentiment", market_sentiment_agent)
    workflow.add_node("investment_advisor", investment_advisor_agent)
    workflow.add_node("report_generator", report_generator_agent)
    
    workflow.set_entry_point("market_intel")
    
    # FAN-OUT: Run News & Corp Research in Parallel
    workflow.add_edge("market_intel", "news_retrieval")
    workflow.add_edge("market_intel", "corporate_research")
    
    #Financials and Sentiment can be executed in parallel
    workflow.add_edge(["news_retrieval", "corporate_research"], "financial_analysis")
    workflow.add_edge(["news_retrieval", "corporate_research"], "market_sentiment")
    
    #FAN-IN: Wait for both Financial and Sentiment to finish before Advising
    workflow.add_edge(["financial_analysis", "market_sentiment"], "investment_advisor")
    
    #Finish
    workflow.add_edge("investment_advisor", "report_generator")
    workflow.add_edge("report_generator", END)
    
    return workflow.compile()

def run_analysis_workflow(ticker: str, analysis_type: str, horizon: str) -> GraphState:
    app = build_workflow()
    initial_state = GraphState(
        ticker=ticker,
        analysis_type=analysis_type,
        horizon=horizon,
        historical_data="",
        intraday_data="",
        financial_metrics={},
        technical_indicators={},
        news_data=[],
        corporate_data=[],
        corporate_research=None,
        financial_analysis=None,
        sentiment_analysis=None,
        investment_recommendation=None,
        generated_report_md="",
        report_path_md="",
        report_path_pdf="",
        report_path_json="",
        error=None
    )
    
    final_state = app.invoke(initial_state)
    return final_state