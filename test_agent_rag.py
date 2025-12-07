#!/usr/bin/env python
"""Quick test of the agent with RAG."""

from app.agent import run_agent

print("Testing agent with RAG...\n")

queries = [
    "What companies are in the Technology sector?",
    "Tell me about Daniel Massey",
    "What deals are closing soon?",
    "Who are the top contacts at Acme Corp?"
]

for q in queries:
    print(f"You: {q}")
    try:
        answer = run_agent(q)
        print(f"Agent: {answer}\n")
    except Exception as e:
        print(f"Error: {e}\n")
