"""LangGraph-based agent for querying ProspectIQ demo data using Groq LLM."""

import os
import json
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .data_loader import DataStore

# Load environment variables
load_dotenv()

# Initialize data store
data_store = DataStore(data_path=os.getenv("DATA_PATH", "output"))

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"  # or another Groq-available model
)


class AgentState(BaseModel):
    """State object for the LangGraph agent."""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""


def process_query(state: AgentState) -> AgentState:
    """Process a user query by interacting with the LLM and data store."""
    user_message = state.messages[-1]["content"] if state.messages else ""

    # Build a rich system prompt with data context
    summary = data_store.get_summary()
    system_prompt = f"""You are a helpful assistant for querying ProspectIQ sales data.
You have access to a dataset with:
- {summary['total_companies']} companies across industries: {', '.join(summary['industries'])}
- {summary['total_contacts']} contacts
- {summary['total_reps']} sales reps (tiers: {', '.join(summary['rep_tiers'])})
- {summary['total_deals']} deals in stages: {', '.join(summary['deal_stages'])}
- {summary['total_emails']} emails
- {summary['total_meetings']} meetings

When a user asks a question, use the data available to provide accurate, contextual answers.
Always be conversational and helpful."""

    # Prepare messages for the LLM
    messages = [SystemMessage(content=system_prompt)]
    for msg in state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        if role == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append({"role": "assistant", "content": msg["content"]})

    # Call Groq LLM
    response = llm.invoke(messages)
    answer = response.content

    # Store in state
    state.messages.append({"role": "assistant", "content": answer})
    state.final_answer = answer
    state.context = {"summary": summary}

    return state


def retrieve_data(state: AgentState) -> AgentState:
    """Optionally retrieve relevant data based on the query."""
    user_message = state.messages[-1]["content"] if state.messages else ""

    # Simple heuristic: search for companies or contacts if mentioned
    if "company" in user_message.lower():
        results = data_store.search_companies(user_message)
        state.context["search_results"] = results[:5]  # Top 5 results

    if "contact" in user_message.lower():
        results = data_store.search_contacts(user_message)
        state.context["search_results"] = results[:5]

    return state


def build_agent_graph():
    """Build the LangGraph agent workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve", retrieve_data)
    graph.add_node("process", process_query)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Add edges
    graph.add_edge("retrieve", "process")
    graph.add_edge("process", END)

    return graph.compile()


def run_agent(user_query: str) -> str:
    """Run the agent with a user query and return the response."""
    agent = build_agent_graph()
    initial_state = AgentState(messages=[{"role": "user", "content": user_query}])
    result = agent.invoke(initial_state)
    return result.final_answer
