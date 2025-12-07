#!/usr/bin/env python
"""Comprehensive demo of RAG agent with various query types."""

from app.agent import run_agent

def demo():
    print("=" * 70)
    print("ProspectIQ RAG Agent Demo — Semantic Search over Generated Data")
    print("=" * 70)
    print()
    
    test_queries = [
        ("Industry search", "What companies are in Healthcare?"),
        ("Contact lookup", "Tell me about Daniel Massey"),
        ("Deal status", "What deals are in the Proposal stage?"),
        ("Company search", "Who are the contacts at Nguyen-Allen?"),
        ("General question", "How many records do we have?"),
        ("Vague query", "Show me some Technology companies and their people"),
    ]
    
    for query_type, query in test_queries:
        print(f"\n[{query_type}]")
        print(f"Q: {query}")
        print("-" * 70)
        try:
            answer = run_agent(query)
            # Truncate very long answers for display
            if len(answer) > 500:
                print(answer[:500] + "\n... (truncated)")
            else:
                print(answer)
        except Exception as e:
            print(f"❌ Error: {e}")
        print()
    
    print("=" * 70)
    print("✓ RAG Agent Demo Complete!")
    print("=" * 70)
    print("\nNotes:")
    print("- All queries used semantic search (TF-IDF vectorization)")
    print("- No hardcoded rules or regex patterns")
    print("- To enable LLM responses, add OPENAI_API_KEY or GROQ_API_KEY to .env")
    print("- Results are automatically formatted by type (companies/contacts/deals/emails)")

if __name__ == "__main__":
    demo()
