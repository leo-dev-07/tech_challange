"""RAG (Retrieval-Augmented Generation) module for semantic search over ProspectIQ data.

Uses sentence-transformers for embeddings and FAISS for fast vector search.
This replaces hardcoded keyword matching with semantic similarity.

If HuggingFace model download fails (network/auth issue), falls back to
a simple TF-IDF vectorizer approach for basic semantic search.
"""

import os
import json
from typing import List, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

HAS_RAG = HAS_TRANSFORMERS or HAS_SKLEARN
SentenceTransformer = SentenceTransformer if HAS_TRANSFORMERS else None
faiss = faiss if HAS_TRANSFORMERS else None


class RAGStore:
    """Semantic search store for ProspectIQ data using embeddings + FAISS or TF-IDF fallback."""

    def __init__(self, data_store, embedding_model_name: str = "all-MiniLM-L6-v2", cache_embeddings: bool = True):
        """
        Args:
            data_store: DataStore instance with companies, contacts, deals, emails, meetings.
            embedding_model_name: HuggingFace sentence-transformers model identifier.
            cache_embeddings: If True, save/load embeddings to/from disk to avoid re-embedding.
        """
        if not HAS_RAG:
            raise ImportError("RAG requires 'sentence-transformers' or 'scikit-learn'. Install via: pip install sentence-transformers faiss-cpu")

        self.data_store = data_store
        self.cache_dir = "embeddings_cache"
        self.cache_embeddings = cache_embeddings
        self.use_transformers = HAS_TRANSFORMERS
        self.model = None
        self.vectorizer = None
        self.tfidf_matrix = None

        # Try to use sentence-transformers; fall back to TF-IDF
        if HAS_TRANSFORMERS:
            try:
                # Use local model name with 'sentence-transformers/' prefix if not already present
                model_name = embedding_model_name
                if not model_name.startswith("sentence-transformers/"):
                    model_name = f"sentence-transformers/{model_name}"
                print(f"Loading embedding model '{model_name}' (this may take a moment)...")
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"[OK] Loaded sentence-transformers model. Embedding dim: {self.embedding_dim}")
                self.use_transformers = True
            except Exception as e:
                print(f"[FALLBACK] Sentence-transformers failed ({type(e).__name__}). Using TF-IDF instead...")
                self.use_transformers = False
                self.model = None

        # If transformers failed, use TF-IDF
        if not self.use_transformers:
            if not HAS_SKLEARN:
                raise ImportError("Fallback TF-IDF requires 'scikit-learn'. Install via: pip install scikit-learn")
            print("[INFO] Using TF-IDF vectorizer for semantic search...")

        # Prepare documents and build index
        self._build_index()

    def _build_index(self):
        """Build search index (FAISS or TF-IDF) from all dataset records."""
        # Collect all documents (each record is a document with metadata)
        self.documents = []
        doc_texts = []

        # Add companies
        for comp in self.data_store.companies:
            revenue = comp.get('annual_revenue_usd') or 0
            doc_text = f"Company: {comp.get('name')} in {comp.get('industry')} with {comp.get('employee_count')} employees. Revenue: ${revenue:,.0f}"
            self.documents.append({"type": "company", "data": comp, "text": doc_text})
            doc_texts.append(doc_text)

        # Add contacts
        for contact in self.data_store.contacts:
            doc_text = f"Contact: {contact.get('full_name')}, {contact.get('title')} at {contact.get('seniority')} level. Email: {contact.get('email')}"
            self.documents.append({"type": "contact", "data": contact, "text": doc_text})
            doc_texts.append(doc_text)

        # Add deals
        for deal in self.data_store.deals:
            amount = deal.get('amount_usd') or 0
            stage = deal.get('stage') or "Unknown"
            close_date = deal.get('expected_close_date') or "TBD"
            doc_text = f"Deal: ${amount:,.0f} in {stage} stage with expected close on {close_date}"
            self.documents.append({"type": "deal", "data": deal, "text": doc_text})
            doc_texts.append(doc_text)

        # Add emails (as brief snippets, limited to avoid huge index)
        for email in self.data_store.emails[:500]:
            subject = email.get('subject') or "No subject"
            sent_date = email.get('date_sent') or "Unknown date"
            sentiment = email.get('sentiment') or "Neutral"
            doc_text = f"Email: {subject} on {sent_date}. Sentiment: {sentiment}"
            self.documents.append({"type": "email", "data": email, "text": doc_text})
            doc_texts.append(doc_text)

        print(f"Indexing {len(self.documents)} documents...")

        if self.use_transformers:
            # Use FAISS with sentence-transformers
            embeddings = self.model.encode(doc_texts, show_progress_bar=False)
            embeddings = embeddings.astype(np.float32)
            self.doc_embeddings = embeddings

            print(f"Building FAISS index with dimension {embeddings.shape[1]}...")
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            print(f"[OK] Index built with {self.index.ntotal} vectors.")
        else:
            # Use TF-IDF with sklearn
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
            self.tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
            print(f"[OK] TF-IDF index built with {self.tfidf_matrix.shape[0]} documents.")

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar documents for a query."""
        if self.use_transformers:
            # FAISS search
            query_embedding = self.model.encode(query).astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(query_embedding, top_k)
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        "type": doc["type"],
                        "text": doc["text"],
                        "data": doc["data"]
                    })
        else:
            # TF-IDF search
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(-similarities)[:top_k]
            results = []
            for idx in top_indices:
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        "type": doc["type"],
                        "text": doc["text"],
                        "data": doc["data"]
                    })

        return results

    def retrieve_context_snippets(self, query: str, max_items: int = 6) -> Dict[str, Any]:
        """Retrieve relevant snippets organized by type (companies, contacts, deals, emails)."""
        results = self.search(query, top_k=max_items * 2)  # Get extra to filter by type

        snippets = {"companies": [], "contacts": [], "deals": [], "emails": []}

        for result in results:
            doc_type = result["type"]
            data = result["data"]

            if doc_type == "company" and len(snippets["companies"]) < max_items:
                snippets["companies"].append({
                    "company_id": data.get("company_id"),
                    "name": data.get("name"),
                    "industry": data.get("industry"),
                    "employee_count": data.get("employee_count"),
                    "annual_revenue_usd": data.get("annual_revenue_usd")
                })
            elif doc_type == "contact" and len(snippets["contacts"]) < max_items:
                snippets["contacts"].append({
                    "contact_id": data.get("contact_id"),
                    "full_name": data.get("full_name"),
                    "title": data.get("title"),
                    "seniority": data.get("seniority"),
                    "email": data.get("email")
                })
            elif doc_type == "deal" and len(snippets["deals"]) < max_items:
                snippets["deals"].append({
                    "deal_id": data.get("deal_id"),
                    "company_id": data.get("company_id"),
                    "amount_usd": data.get("amount_usd"),
                    "stage": data.get("stage"),
                    "expected_close_date": data.get("expected_close_date")
                })
            elif doc_type == "email" and len(snippets["emails"]) < max_items:
                snippets["emails"].append({
                    "email_id": data.get("email_id"),
                    "subject": data.get("subject"),
                    "date_sent": data.get("date_sent"),
                    "sentiment": data.get("sentiment")
                })

        return snippets
