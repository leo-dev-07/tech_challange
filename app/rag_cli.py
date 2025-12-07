"""Simple RAG CLI for querying ProspectIQ demo data.

Usage:
  python -m app.rag_cli

Features:
- Loads data from `output/` via `DataStore`.
- Initializes `RAGStore` (embeddings or TF-IDF fallback) for semantic search.
- Detects aggregation queries (how many, count, total) and computes them across the full dataset.
- For non-aggregation queries, returns top snippets (companies, contacts, deals, emails).
- Lightweight, self-contained CLI (no LLM dependence required).
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

from .data_loader import DataStore
from .rag import RAGStore, HAS_RAG

# Load env
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "output")


def is_aggregation_query(query: str) -> bool:
    q = query.lower()
    keywords = ["how many", "count", "total", "sum", "average", "avg", "min", "max", "number of"]
    return any(k in q for k in keywords)


def analyze_full_dataset(query: str, ds: DataStore) -> str:
    q = query.lower()
    companies = ds.companies
    contacts = ds.contacts
    deals = ds.deals
    emails = ds.emails
    meetings = ds.meetings

    # Specific role counts
    if "manager" in q:
        managers = [c for c in contacts if ("manager" in c.get("title", "").lower() or c.get("seniority", "").lower() == "manager")]
        return f"There are {len(managers)} Managers in the contacts database."
    if "director" in q:
        directors = [c for c in contacts if ("director" in c.get("title", "").lower() or c.get("seniority", "").lower() == "director")]
        return f"There are {len(directors)} Directors in the contacts database."
    if "contact" in q:
        return f"There are {len(contacts)} contacts in the database."
    if "company" in q or "companies" in q:
        return f"There are {len(companies)} companies in the database."
    if "deal" in q:
        return f"There are {len(deals)} deals in the database."
    if "email" in q:
        return f"There are {len(emails)} emails in the database."
    if "meeting" in q:
        return f"There are {len(meetings)} meetings in the database."

    # Fallback summary
    parts = []
    if companies:
        parts.append(f"{len(companies)} companies")
    if contacts:
        parts.append(f"{len(contacts)} contacts")
    if deals:
        parts.append(f"{len(deals)} deals")
    if emails:
        parts.append(f"{len(emails)} emails")
    return "I found: " + ", ".join(parts) + "."


def format_snippets(snippets: Dict[str, Any]) -> str:
    parts = []
    if snippets.get("companies"):
        comps = snippets["companies"]
        parts.append(f"Companies ({len(comps)}): " + ", ".join([c.get("name") for c in comps]))
    if snippets.get("contacts"):
        conts = snippets["contacts"]
        lines = []
        for c in conts:
            lines.append(f"{c.get('full_name')} — {c.get('title')} ({c.get('seniority')})")
        parts.append("Contacts:\n  " + "\n  ".join(lines))
    if snippets.get("deals"):
        deals = snippets["deals"]
        parts.append(f"Deals ({len(deals)}): " + ", ".join([str(d.get('amount_usd', 0)) for d in deals]))
    if snippets.get("emails"):
        emails = snippets["emails"]
        parts.append(f"Emails ({len(emails)}): " + ", ".join([e.get('subject', 'No subject') for e in emails]))
    return "\n\n".join(parts) if parts else "No matching records found."


def main():
    print("=" * 60)
    print("ProspectIQ RAG CLI")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print("Type 'exit' to quit. Type 'help' for commands.\n")

    ds = DataStore(data_path=DATA_PATH)

    rag = None
    if HAS_RAG:
        try:
            rag = RAGStore(ds, embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        except Exception as e:
            print(f"[Warning] RAG initialization failed: {e}. TF-IDF fallback may still work if scikit-learn is installed.")
    else:
        print("[Info] sentence-transformers not available; RAG will use TF-IDF if scikit-learn is present.")

    while True:
        try:
            q = input("Query: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if q.lower() in {"help", "commands"}:
                print("Commands: help, exit. Ask natural language queries (e.g., 'How many managers?','Show SaaS companies')")
                continue

            # Aggregation queries -> full dataset
            if is_aggregation_query(q):
                resp = analyze_full_dataset(q, ds)
                print("\n" + resp + "\n")
                continue

            # Use RAG to retrieve snippets
            if rag is None:
                print("[Error] RAG not initialized and no TF-IDF available. Install sentence-transformers or scikit-learn.")
                continue

            snippets = rag.retrieve_context_snippets(q, max_items=6)
            output = format_snippets(snippets)
            print("\n" + output + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
