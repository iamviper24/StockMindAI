from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- CORPORATE RESEARCH OUTPUT ---
class CorporateResearchOutput(BaseModel):
    management_summary: str = Field(description="Summary of management commentary.")
    quarterly_highlights: str = Field(description="Key quarterly earnings highlights.")
    annual_highlights: str = Field(description="Key annual report highlights.")
    future_guidance: str = Field(description="Future financial and business guidance.")
    expansion_strategy: str = Field(description="Details on market expansion and strategy.")
    acquisitions: str = Field(description="Recent or planned acquisitions.")
    capital_allocation: str = Field(description="CapEx and capital allocation strategy.")
    dividend_policy: str = Field(description="Dividend announcements and policy.")
    institutional_activity: str = Field(description="Institutional ownership and activity.")
    management_risks: str = Field(description="Management identified risks.")
    press_release_summary: str = Field(description="Summary of recent press releases.")
    investor_presentation_summary: str = Field(description="Highlights from investor presentations.")
    filings_summary: str = Field(description="Summary of exchange filings (NSE/BSE/SEC).")
    document_confidence: int = Field(description="Confidence score in the official documents (0-100).")

# --- FINANCIAL AI OUTPUTS ---
class FinancialAnalysisOutput(BaseModel):
    summary: str = Field(description="Plain English explanation of financial health.")
    quarterly_comparison: str = Field(description="Comparison of current vs previous quarter.")
    revenue_analysis: str = Field(description="Analysis of revenue and growth.")
    profitability_analysis: str = Field(description="Analysis of margins and net profit.")
    debt_and_cash: str = Field(description="Analysis of debt, cash flow, and financial risk.")
    financial_confidence: int = Field(description="Score from 0 to 100 representing confidence in financial health.")
    financial_risk: int = Field(description="Score from 0 to 100 representing financial risk.")

class SentimentAnalysisOutput(BaseModel):
    overall_sentiment: str = Field(description="Bullish, Bearish, or Neutral.")
    news_summary: str = Field(description="Summary of key news drivers.")
    market_perception: str = Field(description="How the market currently perceives the company.")
    news_confidence: int = Field(description="Score from 0 to 100 on sentiment confidence.")
    news_risk: int = Field(description="Score from 0 to 100 on news/sentiment risk.")

class InvestmentRecommendationOutput(BaseModel):
    recommendation: str = Field(description="Strong Buy, Buy, Hold, Sell, or Strong Sell.")
    target_horizon: str = Field(description="Recommended investment horizon (e.g., 6-12 Months).")
    bull_case: str = Field(description="The optimistic scenario.")
    bear_case: str = Field(description="The pessimistic scenario.")
    overall_confidence: int = Field(description="Score from 0 to 100.")
    overall_risk: int = Field(description="Score from 0 to 100.")
    key_reasons: List[str] = Field(description="3 to 5 concise reasons for the recommendation.")
    factors_weight: Dict[str, int] = Field(description="Percentage contribution (sum=100) of: Financial Health, Technical Analysis, News Sentiment, Valuation, Market Trend.")

class ReportGenerationOutput(BaseModel):
    markdown_report: str = Field(description="Complete investment report in Markdown format.")

# --- LANGGRAPH STATE ---
class GraphState(TypedDict):
    ticker: str
    analysis_type: str
    horizon: str
    
    # Raw Data
    historical_data: str 
    intraday_data: str   
    financial_metrics: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    news_data: List[Dict[str, str]]
    corporate_data: List[Dict[str, str]] # Raw Firecrawl extracts
    
    # Analyses
    corporate_research: Optional[CorporateResearchOutput]
    financial_analysis: Optional[FinancialAnalysisOutput]
    sentiment_analysis: Optional[SentimentAnalysisOutput]
    investment_recommendation: Optional[InvestmentRecommendationOutput]
    
    # Output
    generated_report_md: str
    report_path_md: str
    report_path_pdf: str
    report_path_json: str
    error: Optional[str]