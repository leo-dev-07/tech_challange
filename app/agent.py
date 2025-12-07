"""RAG-based agent for querying ProspectIQ demo data.

This module uses Retrieval-Augmented Generation (RAG) with semantic search
to retrieve relevant data snippets and passes them to an LLM (Groq/OpenAI) 
or formats them directly if no LLM is available.

No hardcoded rules — all queries are handled via RAG + LLM.
Incorporates user feedback to improve response quality over time.
"""

import os
import json
from typing import Any, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .data_loader import DataStore
from .rag import RAGStore, HAS_RAG
from .feedback import FeedbackStore
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

# Initialize feedback store for learning from user interactions
feedback_store = FeedbackStore()

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


def _check_feedback_for_improvements(user_message: str) -> Dict[str, Any]:
    """Check if there's user feedback that can improve the response.
    
    Returns:
        Dict with 'should_avoid' and 'corrections' keys
    """
    improvements = {
        "should_avoid": feedback_store.should_avoid_pattern(user_message),
        "corrections": feedback_store.get_corrections_for_pattern(user_message),
        "common_issues": dict(feedback_store.get_common_issues(3))
    }
    return improvements


def _format_snippets_as_response(snippets: Dict[str, Any], feedback_improvements: Dict[str, Any] = None) -> str:
    """Format RAG-retrieved snippets into a natural, conversational response.
    
    Generates varied, human-like explanations with closing questions tailored
    to the type of data retrieved (like a teacher explaining to a student).
    
    Args:
        snippets: Retrieved data snippets
        feedback_improvements: Optional feedback data to inform response style
    """
    has_results = any(len(v) > 0 for v in snippets.values())
    
    if not has_results:
        return "I couldn't find any matching records in the database. Try asking about companies, contacts, deals, or other information."
    
    response_parts = []
    
    # Determine what type of data is dominant
    company_count = len(snippets.get("companies", []))
    contact_count = len(snippets.get("contacts", []))
    deal_count = len(snippets.get("deals", []))
    email_count = len(snippets.get("emails", []))
    
    # Dynamic opening based on primary data type
    total_count = sum([company_count, contact_count, deal_count, email_count])
    
    # COMPANIES section
    if company_count > 0:
        companies = snippets["companies"]
        if company_count == 1:
            c = companies[0]
            response_parts.append(f"I found a company: **{c['name']}**. It's a {c['industry']} company with {c['employee_count']} employees.")
        else:
            # Varied opening sentences for multiple companies
            openings = [
                f"There are {company_count} companies matching your search.",
                f"I found {company_count} companies in our database.",
                f"Let me tell you about {company_count} companies we have.",
            ]
            response_parts.append(openings[company_count % len(openings)])
            
            # List companies
            company_names = [f"**{c['name']}**" for c in companies]
            response_parts.append(f"They are: {', '.join(company_names)}.")
            
            # Add industry insight
            industries = list(set(c['industry'] for c in companies))
            if len(industries) == 1:
                response_parts.append(f"All operate in the {industries[0]} sector.")
            else:
                response_parts.append(f"They're spread across {len(industries)} industries: {', '.join(industries)}.")
            
            # Employee range context
            employee_counts = [c['employee_count'] for c in companies]
            min_emp = min(employee_counts)
            max_emp = max(employee_counts)
            avg_emp = int(sum(employee_counts) / len(employee_counts))
            if min_emp == max_emp:
                response_parts.append(f"Each has {min_emp} employees.")
            else:
                response_parts.append(f"They range from {min_emp} to {max_emp} employees in size (averaging {avg_emp}).")
    
    # CONTACTS section
    if contact_count > 0:
        contacts = snippets["contacts"]
        if company_count == 0:  # Only show intro if companies weren't already covered
            if contact_count == 1:
                c = contacts[0]
                response_parts.append(f"\nI found a contact: **{c['full_name']}** — a {c['title']} at {c['seniority']} level.")
            else:
                response_parts.append(f"\nThere are {contact_count} relevant contacts:")
        else:
            response_parts.append(f"\nI also identified {contact_count} key contacts:")
        
        # Show first 2-3 contacts with details
        for i, c in enumerate(contacts[:3]):
            response_parts.append(f"  • **{c['full_name']}** — {c['title']} ({c['seniority'].lower()} level)")
        
        if contact_count > 3:
            response_parts.append(f"  • ... and {contact_count - 3} more contacts")
    
    # DEALS section
    if deal_count > 0:
        deals = snippets["deals"]
        total_value = sum(d.get('amount_usd') or 0 for d in deals)
        
        if deal_count == 1:
            d = deals[0]
            response_parts.append(f"\nThere's 1 deal: **{d.get('description', 'Unnamed deal')}** ({d.get('stage', 'Unknown')} stage) worth ${d.get('amount_usd', 0) or 0:,.0f}.")
        else:
            response_parts.append(f"\nRegarding deals: I found {deal_count} opportunities totaling **${total_value:,.0f}**.")
            
            # Stage breakdown
            stages = {}
            for d in deals:
                stage = d.get('stage', 'Unknown')
                stages[stage] = stages.get(stage, 0) + 1
            
            if len(stages) == 1:
                stage_name = list(stages.keys())[0]
                response_parts.append(f"All are currently in the **{stage_name}** stage.")
            else:
                stage_breakdown = ", ".join([f"{count} in {stage}" for stage, count in sorted(stages.items())])
                response_parts.append(f"They're distributed as: {stage_breakdown}.")
    
    # EMAILS section
    if email_count > 0:
        emails = snippets["emails"]
        response_parts.append(f"\nI also found {email_count} related email(s)")
        
        # Sentiment breakdown if available
        sentiments = {}
        for e in emails:
            sentiment = e.get("sentiment", "Neutral")
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
        
        if sentiments:
            if len(sentiments) == 1:
                sentiment_name = list(sentiments.keys())[0]
                response_parts.append(f"with {sentiment_name.lower()} sentiment.")
            else:
                sentiment_str = ", ".join([f"{count} {s.lower()}" for s, count in sorted(sentiments.items())])
                response_parts.append(f"with this sentiment breakdown: {sentiment_str}.")
    
    # Add a contextual closing question
    if company_count > 0 and deal_count > 0:
        closing_questions = [
            "Would you like to explore the deal pipeline for these companies?",
            "Should we dive deeper into the deal stages or company details?",
            "Want to analyze their sales potential or growth trends?"
        ]
    elif company_count > 0:
        closing_questions = [
            "Would you like to know more about their business activities or deals?",
            "Should I show you the contacts or deals associated with these companies?",
            "Interested in learning about their deal pipeline?"
        ]
    elif contact_count > 0:
        closing_questions = [
            "Would you like to know more about their companies or departments?",
            "Should I show you what deals they're working on?",
            "Want to see their communication history?"
        ]
    else:
        closing_questions = [
            "Is there anything else you'd like to know?",
            "Would you like more details about this data?",
            "What else can I help you with?"
        ]
    
    response_parts.append(f"\n{closing_questions[total_count % len(closing_questions)]}")
    
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
    greeting_patterns = {"hi", "hello", "hey", "greetings", "help", "hi there", "hello there", "hey there"}
    if normalized_message in greeting_patterns or normalized_message.startswith(("hi ", "hello ", "hey ")):
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
        feedback_improvements = _check_feedback_for_improvements(user_message)
        return _format_snippets_as_response(snippets, feedback_improvements)
    else:
        return "No matching records found. Try asking about companies, contacts, or other data fields."


def process_query(state: AgentState) -> AgentState:
    """Process query using RAG + LLM (or direct formatting if no LLM available).
    
    Also checks feedback data to avoid patterns users marked as problematic.
    """
    user_message = state.messages[-1]["content"] if state.messages else ""
    
    # Check if similar queries have received negative feedback
    if feedback_store.should_avoid_pattern(user_message):
        corrections = feedback_store.get_corrections_for_pattern(user_message)
        if corrections:
            # Use user's preferred response for this pattern
            answer = corrections[0]  # Use most recent correction
        else:
            # Retrieve context and format with feedback awareness
            snippets = _retrieve_rag_context(user_message)
            feedback_improvements = _check_feedback_for_improvements(user_message)
            answer = _call_llm_with_context(user_message, snippets)
    else:
        snippets = _retrieve_rag_context(user_message)
        answer = _call_llm_with_context(user_message, snippets)
    
    state.messages.append({"role": "assistant", "content": answer})
    state.final_answer = answer
    state.context = {"source": "rag", "feedback_aware": feedback_store.get_feedback_stats()["total_interactions"] > 0}
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
