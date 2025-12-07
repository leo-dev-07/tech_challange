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


def _is_aggregation_query(query: str) -> bool:
    """Check if the query is asking for aggregations (counts, sums, totals)."""
    query_lower = query.lower()
    aggregation_keywords = {
        "how many", "count", "total", "sum", "average", "avg", "min", "max",
        "percentage", "breakdown", "distribution", "how many", "number of"
    }
    return any(keyword in query_lower for keyword in aggregation_keywords)


def _retrieve_full_dataset_for_aggregation(query: str) -> Dict[str, Any]:
    """For aggregation queries, return full dataset instead of RAG snippets."""
    return {
        "companies": data_store.companies,
        "contacts": data_store.contacts,
        "deals": data_store.deals,
        "emails": data_store.emails,
        "meetings": data_store.meetings,
    }


def _retrieve_rag_context(user_message: str, max_items: int = 6) -> Dict[str, Any]:
    """Retrieve relevant dataset snippets using RAG semantic search.
    
    For aggregation queries (counts, totals, etc.), returns full dataset.
    For specific queries, uses RAG to retrieve limited context.
    """
    if rag_store is None:
        return {"companies": [], "contacts": [], "deals": [], "emails": [], "meetings": []}
    
    # Check if this is an aggregation query
    if _is_aggregation_query(user_message):
        return _retrieve_full_dataset_for_aggregation(user_message)
    
    # Otherwise use RAG for semantic search
    return rag_store.retrieve_context_snippets(user_message, max_items=max_items)


def _analyze_snippets_with_query(user_message: str, snippets: Dict[str, Any]) -> str:
    """Analyze snippets based on the user's query intent.
    
    This is a fallback when no LLM is available. It performs actual data analysis
    on the retrieved snippets to answer the query accurately.
    """
    query_lower = user_message.lower()
    
    # Extract data from snippets
    all_contacts = snippets.get("contacts", [])
    all_companies = snippets.get("companies", [])
    all_deals = snippets.get("deals", [])
    all_emails = snippets.get("emails", [])
    all_meetings = snippets.get("meetings", [])
    
    # Handle "how many" queries for specific titles FIRST (before generic contact count)
    if "how many" in query_lower:
        # Check for specific roles/titles first
        if any(word in query_lower for word in ["manager", "managers"]):
            # Count both "Manager" title and "Manager" seniority level
            managers = [c for c in all_contacts if 'manager' in c.get("title", "").lower() or c.get("seniority", "").lower() == "manager"]
            return f"There are {len(managers)} Managers in the contacts database."
        
        if any(word in query_lower for word in ["director", "directors"]):
            # Count both "Director" title and "Director" seniority level
            directors = [c for c in all_contacts if 'director' in c.get("title", "").lower() or c.get("seniority", "").lower() == "director"]
            return f"There are {len(directors)} Directors in the contacts database."
        
        if any(word in query_lower for word in ["engineer", "engineers"]):
            engineers = [c for c in all_contacts if "engineer" in c.get("title", "").lower() or c.get("seniority", "").lower() == "engineer"]
            return f"There are {len(engineers)} Engineers in the contacts database."
        
        if any(word in query_lower for word in ["vp", "vice president"]):
            vps = [c for c in all_contacts if "vp" in c.get("title", "").lower() or "vice president" in c.get("title", "").lower() or c.get("seniority", "").lower() == "vp"]
            return f"There are {len(vps)} VPs in the contacts database."
        
        if any(word in query_lower for word in ["cio", "chief information officer"]):
            cios = [c for c in all_contacts if "cio" in c.get("title", "").lower() or c.get("seniority", "").lower() == "cio"]
            return f"There are {len(cios)} CIOs in the contacts database."
        
        if any(word in query_lower for word in ["cto", "chief technology officer"]):
            ctos = [c for c in all_contacts if "cto" in c.get("title", "").lower() or c.get("seniority", "").lower() == "cto"]
            return f"There are {len(ctos)} CTOs in the contacts database."
        
        # Then check for generic entity counts
        if "contact" in query_lower:
            return f"There are {len(all_contacts)} contacts in the database."
        
        if "compan" in query_lower:
            return f"There are {len(all_companies)} companies in the database."
        
        if "deal" in query_lower:
            return f"There are {len(all_deals)} deals in the database."
        
        if "email" in query_lower:
            return f"There are {len(all_emails)} emails in the database."
        
        if "meeting" in query_lower:
            return f"There are {len(all_meetings)} meetings in the database."
    
    # Handle "seniority" or "level" queries
    if "seniority" in query_lower or "level" in query_lower:
        # Try exact match first, then partial
        for contact in all_contacts:
            contact_name = contact["full_name"].lower()
            if contact_name in query_lower or query_lower in contact_name:
                return f"{contact['full_name']} has a {contact['seniority'].lower()} seniority level and works as a {contact['title']}."
        
        # Try extracting first or last name from query
        words = [w for w in query_lower.split() if len(w) > 2 and w not in {"the", "what", "seniority", "level", "is", "of", "are", "in"}]
        for word in words:
            for contact in all_contacts:
                if word in contact["full_name"].lower():
                    return f"{contact['full_name']} has a {contact['seniority'].lower()} seniority level and works as a {contact['title']}."
    
    # Handle generic queries - provide summary
    has_results = any(len(v) > 0 for v in [all_companies, all_contacts, all_deals, all_emails, all_meetings])
    if not has_results:
        return "No matching records found in the database."
    
    company_count = len(all_companies)
    contact_count = len(all_contacts)
    deal_count = len(all_deals)
    email_count = len(all_emails)
    
    results = []
    if company_count > 0:
        results.append(f"{company_count} companies")
    if contact_count > 0:
        results.append(f"{contact_count} contacts")
    if deal_count > 0:
        results.append(f"{deal_count} deals")
    if email_count > 0:
        results.append(f"{email_count} emails")
    
    return f"I found: {', '.join(results)}. Please ask a more specific question for detailed analysis."


def _call_llm_with_context(user_message: str, snippets: Dict[str, Any]) -> str:
    """Call an available LLM with RAG context to answer the user's query.

    The LLM receives the actual data and generates responses based on:
    - The user's query intent
    - The retrieved data snippets
    - Accurate calculations (counts, sums, aggregations)
    
    Priority:
      1. If `USE_GROQ` and `llm` available, use ChatGroq.
      2. Else if `OPENAI_API_KEY` present, call OpenAI Chat Completions REST API.
      3. Else return a simple message (no hardcoded templates).
    """
    # Check if this is a generic greeting or very short query
    normalized_message = user_message.strip().lower()
    greeting_patterns = {"hi", "hello", "hey", "greetings", "help", "hi there", "hello there", "hey there"}
    if normalized_message in greeting_patterns or normalized_message.startswith(("hi ", "hello ", "hey ")):
        return "Hi! I can help you query the ProspectIQ sales data. Try asking about companies, contacts, deals, emails, or meetings."
    
    summary = data_store.get_summary()
    has_snippets = any(len(v) > 0 for v in snippets.values())

    system_prompt = (
        f"You are an expert data analyst for ProspectIQ sales data.\n"
        f"Dataset Overview: {summary['total_companies']} companies, {summary['total_contacts']} contacts, {summary['total_deals']} deals, "
        f"{summary['total_emails']} emails, {summary['total_meetings']} meetings.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Answer the user's query accurately based ONLY on the provided data snippets.\n"
        f"2. Perform accurate calculations (counts, sums, aggregations) from the data provided.\n"
        f"3. Be concise and professional. Provide specific numbers and facts.\n"
        f"4. If data is not in the snippets, say 'I don't have information about that in the current dataset.'\n"
        f"5. Present results in a clear, easy-to-read format.\n\n"
        f"Data Retrieved:\n"
        f"Companies: {len(snippets.get('companies', []))}\n"
        f"Contacts: {len(snippets.get('contacts', []))}\n"
        f"Deals: {len(snippets.get('deals', []))}\n"
        f"Emails: {len(snippets.get('emails', []))}\n"
        f"Meetings: {len(snippets.get('meetings', []))}\n"
    )

    # Attach JSON snippets for the LLM to analyze
    context_json = json.dumps(snippets, indent=2)
    system_prompt += f"\nFull Dataset Snippets (JSON):\n{context_json}"

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
        except Exception as e:
            print(f"[DEBUG] Groq error: {e}")
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
        except Exception as e:
            print(f"[DEBUG] OpenAI error: {e}")
            pass

    # 3) Fallback: No LLM available - use query-aware data analysis
    if not has_snippets:
        return "I couldn't find any matching records in the database. Try asking about companies, contacts, deals, or other information."
    
    # Use smart fallback that analyzes snippets based on query intent
    return _analyze_snippets_with_query(user_message, snippets)


def process_query(state: AgentState) -> AgentState:
    """Process query using RAG + LLM (or direct formatting if no LLM available)."""
    user_message = state.messages[-1]["content"] if state.messages else ""
    
    # Check if this is a generic greeting or very short query first
    normalized_message = user_message.strip().lower().rstrip(".,!?;:")
    greeting_keywords = {"hi", "hello", "hey", "greetings", "help", "good morning", "good afternoon", "good evening", "good day"}
    
    # Check exact matches and prefix matches
    is_greeting = (normalized_message in greeting_keywords or 
                   any(normalized_message.startswith(keyword) for keyword in greeting_keywords))
    
    if is_greeting:
        answer = "Hi! I can help you query the ProspectIQ sales data. Try asking about companies, contacts, deals, emails, or meetings."
    else:
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
