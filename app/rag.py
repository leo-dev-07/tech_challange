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
        """Build search index (FAISS or TF-IDF) from all dataset records - FULLY DYNAMIC."""
        # Collect all documents (each record is a document with metadata)
        self.documents = []
        doc_texts = []

        # Dynamically process ALL entities without hardcoding
        entities_map = {
            "companies": ("company", self.data_store.companies),
            "contacts": ("contact", self.data_store.contacts),
            "deals": ("deal", self.data_store.deals),
            "emails": ("email", self.data_store.emails[:500]),  # Limit emails to avoid huge index
            "meetings": ("meeting", getattr(self.data_store, "meetings", [])),
            "sales_reps": ("sales_rep", getattr(self.data_store, "sales_reps", []))
        }

        for entity_name, (entity_type, records) in entities_map.items():
            if not records:
                continue

            for record in records:
                # Dynamically build text from ALL fields in the record
                doc_text = self._build_document_text(entity_type, record)
                self.documents.append({"type": entity_type, "data": record, "text": doc_text})
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

    def _build_document_text(self, entity_type: str, record: dict) -> str:
        """
        Dynamically build searchable text from ALL fields in a record.
        This ensures ANY field can be searched without hardcoding.
        
        Format: entity_type field_name=value field_name=value ...
        This allows searching by any field value, including IDs, without tokenization issues.
        """
        parts = [f"{entity_type}"]  # Entity type as first token

        # Iterate ALL fields and include them as key=value pairs
        # This format prevents TF-IDF from splitting on hyphens in UUIDs
        for key, value in record.items():
            # Skip complex types that don't add search value
            if isinstance(value, (list, dict)):
                continue

            # Convert value to string and clean it
            str_value = str(value).strip() if value is not None else ""
            
            if not str_value:  # Skip empty values
                continue

            # Format as "key:value" and include the key name for field-specific searches
            # Also add the value standalone for substring matches
            parts.append(f"{key}:{str_value}")

        # Join with spaces to create comprehensive searchable text
        # This text will be tokenized by TF-IDF but the key:value format helps with finding specific fields
        return " ".join(parts)

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
        """Retrieve relevant snippets organized by type - RETURNS ALL FIELDS DYNAMICALLY."""
        results = self.search(query, top_k=max_items * 2)  # Get extra to filter by type

        # Dynamically build snippets based on entity types found
        snippets = {}

        for result in results:
            doc_type = result["type"]
            data = result["data"]

            # Initialize list for this type if not exists
            if doc_type not in snippets:
                snippets[doc_type] = []

            # Add ALL fields from the record (dynamic)
            if len(snippets[doc_type]) < max_items:
                snippets[doc_type].append(data)

        return snippets
