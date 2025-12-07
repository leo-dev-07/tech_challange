"""Interactive chat CLI for the ProspectIQ RAG system."""

import os
import sys
from dotenv import load_dotenv
from .main import handle_query
from .data_loader import DataStore
from .rag import RAGStore, HAS_RAG

# Load environment variables
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "output")


def main():
    """Run an interactive chat session with the RAG system."""
    print("\n" + "=" * 60)
    print("ProspectIQ RAG Query App")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}\n")

    # Load data
    ds = DataStore(data_path=DATA_PATH)
    print(f"[OK] Loaded {len(ds.companies)} companies, {len(ds.contacts)} contacts, {len(ds.deals)} deals\n")

    # Initialize RAG
    if not HAS_RAG:
        print("[ERROR] RAG not available. Install sentence-transformers or scikit-learn.\n")
        return

    try:
        print("Initializing RAG semantic search...")
        rag = RAGStore(ds, embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        print("[OK] RAG ready\n")
    except Exception as e:
        print(f"[ERROR] RAG initialization failed: {e}\n")
        return

    print("Commands:")
    print("  Type queries like: 'How many IC contacts?' or 'Show SaaS companies'")
    print("  'exit' to quit\n")

    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            result = handle_query(query, ds, rag)
            print(f"\n{result}\n")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
