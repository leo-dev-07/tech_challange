"""Simple RAG app for querying ProspectIQ data.

Interactive CLI with semantic search (RAG) and LLM-based responses.
- Queries data from output/ (companies, contacts, deals, emails, meetings)
- Uses LLM to generate natural, context-aware responses (no hardcoded templates)
- Falls back to raw data display if LLM not available
- Dynamically analyzes query to provide comprehensive statistical context
"""

import os
import json
from collections import Counter, defaultdict
from dotenv import load_dotenv
from .data_loader import DataStore
from .rag import RAGStore, HAS_RAG

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "output")


def get_llm_response(query: str, context: str) -> str:
    """Call LLM to generate natural response based on query and data context."""
    # Try Groq first
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            
            client = Groq(api_key=groq_key)
            message = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst. Answer the user's query based on the provided data context. Be concise and direct."},
                    {"role": "user", "content": f"Query: {query}\n\nData context:\n{context}"}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=1000,
            )
            return message.choices[0].message.content
        except Exception as e:
            pass  # Silently try next option
    
    # Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import requests
            url = "https://api.openai.com/v1/chat/completions"
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful data analyst. Answer the user's query based on the provided data context. Be concise and direct."},
                    {"role": "user", "content": f"Query: {query}\n\nData context:\n{context}"}
                ],
                "temperature": 0
            }
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices"):
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            pass  # Silently fall through
    
    return None


class QueryAnalyzer:
    """Analyzes queries to extract dimensions, filters, and aggregation intents."""
    
    def __init__(self, ds: DataStore):
        self.ds = ds
        self.query_lower = None
        self.all_values = {}  # Store all unique values from all fields in all entities
        self._build_value_index()
    
    def _build_value_index(self):
        """
        Dynamically discover ALL unique values from ALL fields in ALL records.
        This creates a searchable index of every piece of data without hardcoding.
        """
        # Process all entities dynamically
        entities = {
            "contacts": self.ds.contacts,
            "companies": self.ds.companies,
            "deals": self.ds.deals,
            "emails": self.ds.emails,
            "meetings": getattr(self.ds, "meetings", []),
            "sales_reps": getattr(self.ds, "sales_reps", [])
        }

        # For each entity type, extract all unique values
        for entity_type, records in entities.items():
            if not records:
                continue

            for record in records:
                for field_name, field_value in record.items():
                    # Skip complex types
                    if isinstance(field_value, (list, dict)):
                        continue

                    # Skip None/empty values
                    if field_value is None or str(field_value).strip() == "":
                        continue

                    # Create a normalized key for lookups
                    key = str(field_value).strip().lower()

                    # Store both the original value and which field it came from
                    if key not in self.all_values:
                        self.all_values[key] = {
                            "original": field_value,
                            "fields": set(),
                            "entity_types": set()
                        }

                    self.all_values[key]["fields"].add(field_name)
                    self.all_values[key]["entity_types"].add(entity_type)
    
    def analyze(self, query: str) -> dict:
        """Analyze query to determine what filters and aggregations are needed."""
        self.query_lower = query.lower()
        
        analysis = {
            "is_aggregation": self._is_aggregation_query(),
            "requested_entities": self._identify_entities(),
            "filters": self._identify_filters(),
            "dimensions": self._identify_dimensions_to_include()
        }
        return analysis
    
    def _is_aggregation_query(self) -> bool:
        """Check if query is asking for counts/aggregations."""
        keywords = ["how many", "count", "total", "number", "how much", "statistics", "summary", "distribution"]
        return any(kw in self.query_lower for kw in keywords)
    
    def _identify_entities(self) -> list:
        """Identify which data entities are relevant to the query."""
        entities = []
        if any(word in self.query_lower for word in ["contact", "people", "sales rep", "representative", "person"]):
            entities.append("contacts")
        if any(word in self.query_lower for word in ["compan", "customer", "client", "organization"]):
            entities.append("companies")
        if any(word in self.query_lower for word in ["deal", "opportunity", "sales", "pipeline"]):
            entities.append("deals")
        if any(word in self.query_lower for word in ["email", "message", "communication", "correspondence"]):
            entities.append("emails")
        if any(word in self.query_lower for word in ["meeting", "call", "event", "conference", "meeting_id"]):
            entities.append("meetings")
        if any(word in self.query_lower for word in ["sales rep", "rep", "representative"]):
            entities.append("sales_reps")
        
        return entities if entities else ["contacts", "companies", "deals", "emails", "meetings"]  # Default to all
    
    def _identify_filters(self) -> dict:
        """Identify filters in query by matching against all discovered values."""
        filters = {}
        query_tokens = self.query_lower.split()

        # Check if any discovered value appears in the query
        for token in query_tokens:
            if token in self.all_values:
                value_info = self.all_values[token]
                # Map which fields this value was found in
                for field_name in value_info["fields"]:
                    filters[field_name] = {
                        "value": value_info["original"],
                        "entity_types": list(value_info["entity_types"])
                    }

        return filters
    
    def _identify_dimensions_to_include(self) -> list:
        """Dynamically identify which dimensions appear in retrieved data."""
        # Instead of hardcoding which dimensions to check,
        # we'll let the context builder handle all dimensions present in the data
        return []  # Let StatisticsBuilder discover dimensions from snippets


class StatisticsBuilder:
    """Builds comprehensive statistics context without hardcoding."""
    
    def __init__(self, ds: DataStore):
        self.ds = ds
    
    def build_context(self, query: str, snippets: dict, analyzer: QueryAnalyzer) -> str:
        """Build comprehensive context based on query analysis."""
        analysis = analyzer.analyze(query)
        analysis["snippets"] = snippets  # Add snippets to analysis for dimensional stats
        context_parts = []
        
        # 1. Add retrieved snippets with entity names and counts
        context_parts.append(self._format_snippets(snippets))
        
        # 2. Add aggregation statistics if this is a count/total query
        if analysis["is_aggregation"]:
            context_parts.append(self._build_aggregation_stats(analysis, snippets))
        
        # 3. Add dimensional distributions (now discovered dynamically from snippets)
        dimension_stats = self._build_dimensional_stats(analysis)
        if dimension_stats:
            context_parts.append(dimension_stats)
        
        return "\n\n".join(filter(None, context_parts))
    
    def _format_snippets(self, snippets: dict) -> str:
        """Format retrieved snippets without hardcoding entity types."""
        parts = []
        
        for entity_type, items in snippets.items():
            if items:
                count = len(items)
                total = self._get_entity_total(entity_type)
                header = f"{entity_type.capitalize()} ({count} retrieved, {total} total in database)"
                content = json.dumps(items[:5], indent=2)
                parts.append(f"{header}:\n{content}")
        
        return "\n\n".join(parts) if parts else "No data snippets retrieved."
    
    def _get_entity_total(self, entity_type: str) -> int:
        """Dynamically get total count for any entity type."""
        entity_map = {
            "contacts": len(self.ds.contacts),
            "companies": len(self.ds.companies),
            "deals": len(self.ds.deals),
            "emails": len(self.ds.emails),
            "meetings": len(getattr(self.ds, "meetings", [])),
            "sales_reps": len(getattr(self.ds, "sales_reps", []))
        }
        return entity_map.get(entity_type, 0)
    
    def _build_aggregation_stats(self, analysis: dict, snippets: dict) -> str:
        """Build overall aggregation statistics for ALL entities."""
        parts = ["Database Statistics:"]
        
        # Add totals for all entities
        for entity_type in ["contacts", "companies", "deals", "emails", "meetings", "sales_reps"]:
            total = self._get_entity_total(entity_type)
            if total > 0:
                parts.append(f"- Total {entity_type}: {total}")
        
        return "\n".join(parts)
    
    def _build_dimensional_stats(self, analysis: dict) -> str:
        """
        Dynamically build statistics for ALL dimensions found in retrieved snippets.
        No hardcoding - discovers distributions from actual data.
        """
        parts = []

        for entity_type, items in analysis.get("snippets", {}).items():
            if not items:
                continue

            # Extract all field names and their value distributions
            field_distributions = defaultdict(Counter)

            for item in items:
                for field_name, field_value in item.items():
                    # Skip complex types and IDs
                    if isinstance(field_value, (list, dict)):
                        continue
                    if field_name.endswith("_id") or field_name == "id":
                        continue

                    # Track value distributions for every field
                    field_distributions[field_name][str(field_value)] += 1

            # Build stats for each field
            for field_name, value_counts in field_distributions.items():
                if len(value_counts) > 1:  # Only report if there's actual distribution
                    header = f"{entity_type.capitalize()} - {field_name.replace('_', ' ').title()} Distribution"
                    stats = ", ".join([f"{val}={count}" for val, count in sorted(value_counts.items(), key=lambda x: -x[1])[:5]])
                    parts.append(f"{header}:\n{stats}")

        return "\n\n".join(parts) if parts else ""


def handle_query(query: str, ds: DataStore, rag: RAGStore) -> str:
    """Process query with RAG and LLM for natural response using scalable context building."""
    # Analyze query to understand what the user wants
    analyzer = QueryAnalyzer(ds)
    stats_builder = StatisticsBuilder(ds)
    
    # Retrieve relevant data via RAG
    snippets = rag.retrieve_context_snippets(query, max_items=10)
    
    # Build comprehensive context without hardcoding
    context = stats_builder.build_context(query, snippets, analyzer)
    
    # Get LLM response
    response = get_llm_response(query, context)
    if response:
        return response
    
    # Fallback: show raw data if LLM not available
    return f"Raw data retrieved:\n{context}"


def main():
    print("\n" + "=" * 60)
    print("ProspectIQ RAG Query App")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}\n")

    # Load data
    ds = DataStore(data_path=DATA_PATH)
    print(f"[OK] Loaded {len(ds.companies)} companies, {len(ds.contacts)} contacts, {len(ds.deals)} deals\n")

    # Initialize RAG
    rag = None
    if HAS_RAG:
        try:
            print("Initializing RAG semantic search...")
            rag = RAGStore(ds, embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
            print("[OK] RAG ready\n")
        except Exception as e:
            print(f"[ERROR] RAG initialization failed: {e}\n")
            raise
    else:
        print("[ERROR] RAG not available. Install sentence-transformers or scikit-learn.\n")
        raise RuntimeError("RAG required to run this app")

    print("Commands:")
    print("  Type queries like: 'How many IC contacts?' or 'Show SaaS companies'")
    print("  'exit' to quit\n")

    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue
            
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            result = handle_query(query, ds, rag)
            print(f"\n{result}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
