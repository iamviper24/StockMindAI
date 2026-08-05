import os
import json
import requests
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import FAISS_DB_DIR, REPORTS_DIR


from analysis import get_cached_embeddings

def ingest_report_to_faiss(ticker: str, report_text: str):
    if not report_text: return
    
    json_path = REPORTS_DIR / f"{ticker}.json"
    corp_data_string = ""
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cr = data.get("corporate_research")
                if cr:
                    corp_data_string = f"\n\n--- OFFICIAL CORPORATE DOCUMENTS ---\n{json.dumps(cr, indent=2)}"
        except Exception:
            pass
            
    combined_text = report_text + corp_data_string
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(combined_text)
    
    embeddings = get_cached_embeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    db_path = FAISS_DB_DIR / ticker
    vectorstore.save_local(str(db_path))

def perform_live_web_search(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key: return ""
    try:
        res = requests.post(
            "https://api.tavily.com/search", 
            json={"api_key": api_key, "query": query, "search_depth": "basic", "max_results": 3}, 
            timeout=5
        )
        if res.status_code == 200:
            results = res.json().get("results", [])
            return "\n\n".join([f"- {r.get('content')}" for r in results])
    except Exception: pass
    return ""

def chat_with_report(ticker: str, query: str) -> str:
    db_path = FAISS_DB_DIR / ticker
    faiss_context = "No local report cached."
    
    if db_path.exists():
        embeddings = get_cached_embeddings()
        vectorstore = FAISS.load_local(str(db_path), embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(query, k=3)
        faiss_context = "\n\n".join([doc.page_content for doc in docs])
        
    web_query = f"{ticker} stock {query}"
    web_context = perform_live_web_search(web_query)
    if not web_context: web_context = "No real-time web search results retrieved."

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3)
    
    template = """You are a highly intelligent, expert financial advisor and stock analyst answering questions about {ticker}.
    
    Below is the context gathered to help you answer the question. It contains data from a locally generated AI report AND real-time web search results.
    
    --- CACHED REPORT DATA ---
    {faiss_context}
    
    --- LIVE WEB SEARCH DATA ---
    {web_context}
    ---------------------------
    
    Question: {question}
    
    INSTRUCTIONS:
    1. First, try to answer the question using the Report Data or Live Web Data provided above.
    2. If the exact answer is not in the text, DO NOT say "I don't have enough information" or "I don't know."
    3. Instead, use your vast training and general financial knowledge about {ticker} to provide a highly educated, helpful, and logical response.
    4. Keep the answer professional, concise, and direct.
    
    Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["ticker", "faiss_context", "web_context", "question"])
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"ticker": ticker, "faiss_context": faiss_context, "web_context": web_context, "question": query})