# StockMindAI — AI-Powered Stock Research Platform

StockMindAI is a AI stock research platform that combines **market data, technical analysis, financial fundamentals, news, corporate research, sentiment analysis, and investment reasoning** into one multi-agent workflow.

The main idea is to **process and reduce financial information before sending it to the LLM**, balancing analysis quality, token usage, API calls, and response time.

---

## Key Features

-  **Global stock research** across supported markets
-  **Interactive stock charts** with multiple timeframes and chart types
-  **Technical indicators** such as RSI, MACD, SMA, EMA, ADX, ATR, OBV and Bollinger Bands
-  **Multi-source news retrieval** using Yahoo Finance and Tavily
-  **Corporate research** using Firecrawl
-  **Fundamental/financial analysis**
-  **AI sentiment analysis**
-  **Multi-agent investment analysis**
-  **AI-generated research reports**
-  **Follow-up questions using RAG**
-  **Watchlist support**
-  **Caching, filtering, deduplication and parallel processing** for better performance

---

# Architecture

The project is organized around a LangGraph multi-agent workflow:

```text
                    User
                     |
                     v
          Request data for a Stock
                     |
                     v
               Market Intel
                     |
          +----------+----------+
          |                     |
          v                     v
    News Retrieval      Corporate Research
          |                     |
          +----------+----------+
                     |
            +--------+--------+
            |                 |
            v                 v
     Financial Analysis   Sentiment Analysis
            |                 |
            +--------+--------+
                     |
                     v
             Investment Advisor
                     |
                     v
              Report Generator
                     |
                     v
                 Final Report
                     |
                     v
                 FAISS RAG
                     |
                     v
              Follow-up Chat
```

The seven agents are:

1. **Market Intelligence** — retrieves market/financial data and calculates technical indicators.
2. **News Retrieval** — gathers and filters relevant news.
3. **Corporate Research** — retrieves and processes company documents.
4. **Financial Analysis** — evaluates financial health and performance.
5. **Sentiment Analysis** — evaluates news and market sentiment.
6. **Investment Advisor** — combines the different analysis dimensions into a recommendation.
7. **Report Generator** — converts the results into a structured research report.

---

# Main Files

## `app.py`

The main Streamlit application.

Handles:

- UI
- stock selection
- watchlist
- charts
- analysis controls
- displaying AI reports
- follow-up chat
- session state

It acts as the main interface between the user and the backend workflow.

---

## `config.py`

Contains common configuration such as:

- API settings
- directories
- cache locations
- report locations
- FAISS locations
- environment variables

This keeps configuration centralized.

---

## `services.py`

Handles external data sources.

Main responsibilities:

- **yFinance** → prices, company information and financial statements
- **Tavily** → additional news/web search
- **Firecrawl** → corporate/investor information
- market-mover retrieval
- company-name and currency handling

This separates external API interaction from the analysis logic.

---

## `analysis.py`

Contains the main data-processing and analytical utilities.

Important functionality:

### Technical analysis

Calculates indicators such as:

```text
SMA 50 / 200
EMA 20
RSI
MACD
ADX
ATR
OBV
Bollinger Bands
VWAP
Support / Resistance
```

### News processing

Uses:

```text
TF-IDF
+
Cosine Similarity
```

to identify duplicate/similar articles, then limits and compresses the final news context.

### Corporate document processing

Uses:

```text
Source authority
+
Embeddings
+
Cosine similarity
```

to rank and remove redundant documents before sending them to the LLM.

### Visualization

Creates the interactive Plotly stock charts and handles timeframe selection.

---

## `models.py`

Defines Pydantic models used as structured contracts between agents.

Examples include:

```text
GraphState
CorporateResearchOutput
FinancialAnalysisOutput
SentimentAnalysisOutput
InvestmentRecommendationOutput
ReportGenerationOutput
```

This makes LLM outputs more predictable and easier for the next agent to consume.

---

## `graph.py`

Contains the LangGraph workflow.

It:

- defines the agent nodes
- defines dependencies between agents
- manages shared `GraphState`
- executes the complete analysis

The graph allows independent tasks such as news retrieval and corporate research to be processed separately before downstream analysis.

---

## `rag.py`

Implements follow-up question answering.

The generated report is:

```text
Report
  ↓
Text Splitting
  ↓
Embeddings
  ↓
FAISS
```

For a follow-up question:

```text
User Question
      |
      +---------> FAISS
      |              |
      |        Relevant chunks
      |
      +---------> Tavily
                     |
                Fresh web data
                     |
                     v
                   Gemini
                     |
                     v
                  Answer
```

This avoids rerunning the entire seven-agent workflow for every question.

---

## `utils.py`

Contains general utilities and persistence functionality.

Handles:

- JSON loading/saving
- JSON-safe conversion of Pandas/datetime values
- watchlist management

---

# Data Flow

For a selected ticker, the initial workflow roughly follows:

```text
Ticker
  |
  v
Market + Financial Data
  |
  +--> Technical Indicators
  |
  +--> News --> Filtering/Deduplication
  |
  +--> Corporate Documents --> Ranking/Deduplication
                         |
                         v
                 Specialized Analysis
                         |
                         v
                 Investment Advisor
                         |
                         v
                  Final AI Report
```

---

# Timeframe Handling

The project retrieves data at a few useful resolutions rather than making a separate API request for every chart timeframe.

For example:

```text
5Y Daily Data
   |
   +--> 3M
   +--> 6M
   +--> 1Y
   +--> 5Y

1M Hourly Data
   |
   +--> 1W
   +--> 1M

1D 5-minute Data
   |
   +--> 1D
```

This allows timeframe changes to be handled largely through **local slicing**, reducing unnecessary API calls.

---

# Token & Latency Optimization

One of the key design decisions is:

> **Don't send all raw financial information directly to the LLM.**

Instead:

```text
Large Raw Data
      ↓
Python Processing
      ↓
Filtering / Ranking / Deduplication
      ↓
Compact Context
      ↓
LLM Reasoning
```

### Examples

**Market data**

Thousands of price rows → technical indicators and relevant summaries.

**News**

Many articles → deduplication → limited top articles.

**Corporate research**

Many documents → authority ranking → semantic deduplication → selected documents.

**Follow-up questions**

Complete seven-agent workflow → FAISS + limited web search + LLM.

This reduces both **token consumption and unnecessary computation**.

---

# Performance Optimization

The project combines several techniques:

### Caching

Frequently requested data is cached so repeated requests do not always hit external APIs.

### Parallel workflow branches

Independent research tasks can execute separately instead of forcing everything into one long sequential chain.

### Batch/concurrent market retrieval

Market dashboard information can be retrieved efficiently for multiple stocks.

### Local timeframe slicing

Previously retrieved historical data is reused for different chart periods.

### Lightweight RAG follow-ups

Follow-up questions do not require a complete fresh analysis.

---



# Technology Stack

| Category | Technology |
|---|---|
| UI | Streamlit |
| Language | Python |
| Market Data | yFinance |
| Web Search | Tavily |
| Web Crawling | Firecrawl |
| LLM | Google Gemini |
| Agent Workflow | LangGraph |
| Validation | Pydantic |
| Data Processing | Pandas / NumPy |
| Embeddings | HuggingFace |
| Vector Search | FAISS |
| Charts | Plotly |
| Persistence | JSON / Markdown |

---

# Quick Start

```bash
git clone <repository-url>
cd StockMindAI

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create `.env` with the required API keys:

```env
GEMINI_API_KEY=your_key
TAVILY_API_KEY=your_key
FIRECRAWL_API_KEY=your_key
```

Run:

```bash
streamlit run app.py
```

---

# High-Level Design Philosophy

StockMindAI follows a simple principle:

> **Retrieve broadly, process deterministically, reason selectively, and retrieve locally for follow-ups.**

This lets the system provide a relatively comprehensive stock analysis while avoiding the cost and latency of sending every piece of raw financial information to an LLM.

---

## Disclaimer

StockMindAI is an educational/research application. AI-generated analysis and recommendations are not guaranteed predictions and should not be treated as personalized financial advice.
