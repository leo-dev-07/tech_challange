"""RAG-based agent for querying ProspectIQ demo data.

This module uses Retrieval-Augmented Generation (RAG) with semantic search
to retrieve relevant data snippets and passes them to an LLM (Groq/OpenAI) 
or formats them directly if no LLM is available.

No hardcoded rules — all queries are handled via RAG + LLM.
"""

import os
import json
from typing import Any, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .data_loader import DataStore
from .rag import RAGStore, HAS_RAG
import requests

# Optional imports for Groq/langgraph
USE_GROQ = False
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    USE_GROQ = True
except Exception:
    ChatGroq = None
    HumanMessage = None
    SystemMessage = None
    StateGraph = None
    END = None

# Load environment variables
load_dotenv()

# Initialize data store
data_store = DataStore(data_path=os.getenv("DATA_PATH", "output"))

# Initialize RAG store for semantic search
rag_store = None
if HAS_RAG:
    try:
        print("Initializing RAG semantic search...")
        rag_store = RAGStore(data_store, embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    except Exception as e:
        print(f"Error: Could not initialize RAG store: {e}")
        raise

# If Groq is available, initialize the LLM client
llm = None
if USE_GROQ:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"))


class AgentState(BaseModel):
    """State object for the agent."""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""


def _retrieve_rag_context(user_message: str, max_items: int = 6) -> Dict[str, Any]:
    """Retrieve relevant dataset snippets using RAG semantic search."""
    if rag_store is None:
        return {"companies": [], "contacts": [], "deals": [], "emails": []}
    return rag_store.retrieve_context_snippets(user_message, max_items=max_items)


def _format_snippets_as_response(snippets: Dict[str, Any]) -> str:
    """Format RAG-retrieved snippets into a friendly readable response."""
    has_results = any(len(v) > 0 for v in snippets.values())
    
    if not has_results:
        return "I couldn't find any matching records. Try asking about companies, contacts, deals, or other data fields."
    
    response_parts = []
    
    if snippets.get("companies"):
        response_parts.append("**Companies found:**")
        for c in snippets["companies"]:
            response_parts.append(f"  - {c['name']} ({c['industry']}, {c['employee_count']} employees)")
    
    if snippets.get("contacts"):
        response_parts.append("\n**Contacts found:**")
        for c in snippets["contacts"]:
            response_parts.append(f"  - {c['full_name']}: {c['title']} ({c['seniority']})")
    
    if snippets.get("deals"):
        response_parts.append("\n**Deals found:**")
        for d in snippets["deals"]:
            response_parts.append(f"  - ${d['amount_usd']:,.0f} in {d['stage']} stage")
    
    if snippets.get("emails"):
        response_parts.append("\n**Emails found:**")
        for e in snippets["emails"][:3]:
            subject = e.get("subject", "No subject")
            sentiment = e.get("sentiment", "Neutral")
            response_parts.append(f"  - {subject} ({sentiment})")
    
    return "\n".join(response_parts)


def _call_llm_with_context(user_message: str, snippets: Dict[str, Any]) -> str:
    """Call an available LLM with RAG context, or format snippets directly.

    Priority:
      1. If `USE_GROQ` and `llm` available, use ChatGroq.
      2. Else if `OPENAI_API_KEY` present, call OpenAI Chat Completions REST API.
      3. Else format the RAG snippets directly as a response.
    
    For very short/generic queries (like "hello"), provides a helpful response
    without forcing through RAG results.
    """
    # Check if this is a generic greeting or very short query
    normalized_message = user_message.strip().lower()
    if normalized_message in {"hi", "hello", "hey", "hey there", "greetings", "help"}:
        return "Hi! I can help you query the ProspectIQ sales data. Try asking about companies, contacts, deals, emails, or meetings."
    
    summary = data_store.get_summary()
    has_snippets = any(len(v) > 0 for v in snippets.values())

    system_prompt = (
        f"You are a helpful assistant for querying ProspectIQ sales data.\n"
        f"Dataset: {summary['total_companies']} companies, {summary['total_contacts']} contacts, {summary['total_deals']} deals.\n"
        "Use ONLY the provided dataset snippets. Be concise and professional.\n"
    )

    # Attach JSON snippets
    context_json = json.dumps(snippets, indent=2)
    system_prompt += f"\nDataset snippets:\n{context_json}"

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # 1) Try Groq via ChatGroq
    if USE_GROQ and llm is not None:
        try:
            lc_msgs = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            response = llm.invoke(lc_msgs)
            return getattr(response, "content", str(response))
        except Exception:
            pass

    # 2) Try OpenAI Chat Completions via REST
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            payload = {"model": model, "messages": messages, "temperature": 0}
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices"):
                msg = data["choices"][0].get("message", {}).get("content")
                if msg:
                    return msg
        except Exception:
            pass

    # 3) No LLM available — format snippets directly or provide a helpful message
    if has_snippets:
        return _format_snippets_as_response(snippets)
    else:
        return "No matching records found. Try asking about companies, contacts, or other data fields."


def process_query(state: AgentState) -> AgentState:
    """Process query using RAG + LLM (or direct formatting if no LLM available)."""
    user_message = state.messages[-1]["content"] if state.messages else ""
    snippets = _retrieve_rag_context(user_message)
    answer = _call_llm_with_context(user_message, snippets)
    state.messages.append({"role": "assistant", "content": answer})
    state.final_answer = answer
    state.context = {"source": "rag"}
    return state


def build_agent_graph():
    """Build a simple agent graph (or fallback if langgraph is unavailable)."""
    if USE_GROQ and StateGraph is not None:
        graph = StateGraph(AgentState)
        graph.add_node("process", process_query)
        graph.set_entry_point("process")
        graph.add_edge("process", END)
        return graph.compile()

    # Fallback: simple invoke wrapper
    class SimpleAgent:
        def invoke(self, state: AgentState):
            return process_query(state)

    return SimpleAgent()


def run_agent(user_query: str) -> str:
    """Run the agent with a user query and return the response."""
    agent = build_agent_graph()
    initial_state = AgentState(messages=[{"role": "user", "content": user_query}])
    result = agent.invoke(initial_state)
    return result.final_answer
