#!/usr/bin/env python
"""Debug RAG retrieval."""

from app.data_loader import DataStore
from app.rag import RAGStore

ds = DataStore(data_path="output")
rag = RAGStore(ds)

queries = [
    "What companies are in the Technology sector?",
    "Tell me about Daniel Massey",
    "What deals are closing soon?",
    "Who are the top contacts at Acme Corp?"
]

for q in queries:
    print(f"\nQuery: {q}")
    snippets = rag.retrieve_context_snippets(q)
    print(f"  Companies: {len(snippets['companies'])}")
    print(f"  Contacts: {len(snippets['contacts'])}")
    print(f"  Deals: {len(snippets['deals'])}")
    print(f"  Emails: {len(snippets['emails'])}")
    if snippets['companies']:
        print(f"  Sample company: {snippets['companies'][0]['name']}")
