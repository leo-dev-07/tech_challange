#!/usr/bin/env python
"""Quick test of RAG initialization with TF-IDF."""

from app.data_loader import DataStore
from app.rag import RAGStore

print("Testing RAG initialization...")
ds = DataStore(data_path="output")
print(f"DataStore loaded: {len(ds.companies)} companies, {len(ds.contacts)} contacts")

rag = RAGStore(ds)
print("✓ RAG store initialized")

# Test search
results = rag.search("What companies are in technology?", top_k=3)
print(f"\nSearch results for 'technology companies':")
for r in results:
    print(f"  - {r['type']}: {r['text'][:80]}...")
