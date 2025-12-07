"""Simple RAG app for querying ProspectIQ data.

Interactive CLI with semantic search (RAG) and LLM-based responses.
- Queries data from output/ (companies, contacts, deals, emails, meetings)
- Uses LLM to generate natural, context-aware responses (no hardcoded templates)
- Falls back to raw data display if LLM not available
"""

import os
import json
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


def handle_query(query: str, ds: DataStore, rag: RAGStore) -> str:
    """Process query with RAG and LLM for natural response."""
    # Retrieve relevant data via RAG
    snippets = rag.retrieve_context_snippets(query, max_items=10)
    
    # Build context from retrieved data
    context_parts = []
    
    if snippets.get("companies"):
        context_parts.append(f"Companies ({len(snippets['companies'])} retrieved, {len(ds.companies)} total in database):\n" + json.dumps(snippets["companies"][:5], indent=2))
    if snippets.get("contacts"):
        context_parts.append(f"Contacts ({len(snippets['contacts'])} retrieved, {len(ds.contacts)} total in database):\n" + json.dumps(snippets["contacts"][:5], indent=2))
    if snippets.get("deals"):
        context_parts.append(f"Deals ({len(snippets['deals'])} retrieved, {len(ds.deals)} total in database):\n" + json.dumps(snippets["deals"][:5], indent=2))
    if snippets.get("emails"):
        context_parts.append(f"Emails ({len(snippets['emails'])} retrieved, {len(ds.emails)} total in database):\n" + json.dumps(snippets["emails"][:5], indent=2))
    
    # Add full dataset summary for aggregation queries
    if any(kw in query.lower() for kw in ["how many", "count", "total", "number"]):
        context_parts.append(f"\nFull database statistics:\n- Total contacts: {len(ds.contacts)}\n- Total companies: {len(ds.companies)}\n- Total deals: {len(ds.deals)}\n- Total emails: {len(ds.emails)}")
        
        # For seniority/role filtering, add counts
        if "ic" in query.lower():
            ic_count = sum(1 for c in ds.contacts if c.get("seniority", "").lower() == "ic")
            context_parts.append(f"- IC contacts: {ic_count}")
        if "vp" in query.lower():
            vp_count = sum(1 for c in ds.contacts if c.get("seniority", "").lower() == "vp")
            context_parts.append(f"- VP contacts: {vp_count}")
        if "manager" in query.lower():
            manager_count = sum(1 for c in ds.contacts if c.get("seniority", "").lower() == "manager")
            context_parts.append(f"- Manager contacts: {manager_count}")
        if "director" in query.lower():
            director_count = sum(1 for c in ds.contacts if c.get("seniority", "").lower() == "director")
            context_parts.append(f"- Director contacts: {director_count}")
        if "c-level" in query.lower():
            clevel_count = sum(1 for c in ds.contacts if c.get("seniority", "").lower() == "c-level")
            context_parts.append(f"- C-Level contacts: {clevel_count}")
        
        # For email sentiment filtering, add counts
        if "sentiment" in query.lower() or "positive" in query.lower() or "negative" in query.lower() or "neutral" in query.lower():
            positive_count = sum(1 for e in ds.emails if e.get("sentiment", "").lower() == "positive")
            neutral_count = sum(1 for e in ds.emails if e.get("sentiment", "").lower() == "neutral")
            negative_count = sum(1 for e in ds.emails if e.get("sentiment", "").lower() == "negative")
            context_parts.append(f"- Email sentiments: Positive={positive_count}, Neutral={neutral_count}, Negative={negative_count}")
        
        # For company industry filtering, add counts
        if "compan" in query.lower() and "industry" in query.lower():
            industries = {}
            for c in ds.companies:
                ind = c.get("industry", "Unknown")
                industries[ind] = industries.get(ind, 0) + 1
            industry_str = ", ".join([f"{ind}={count}" for ind, count in sorted(industries.items())])
            context_parts.append(f"- Company industries: {industry_str}")
        elif "compan" in query.lower():
            # Check if query mentions a specific industry
            industries = {}
            for c in ds.companies:
                ind = c.get("industry", "Unknown")
                industries[ind] = industries.get(ind, 0) + 1
            for industry in industries.keys():
                if industry.lower() in query.lower():
                    context_parts.append(f"- {industry} companies: {industries[industry]}")
        
        # For deal stage filtering, add counts
        if "deal" in query.lower() and "stage" in query.lower():
            stages = {}
            for d in ds.deals:
                stage = d.get("stage", "Unknown")
                stages[stage] = stages.get(stage, 0) + 1
            stage_str = ", ".join([f"{stage}={count}" for stage, count in sorted(stages.items())])
            context_parts.append(f"- Deal stages: {stage_str}")
        elif "deal" in query.lower():
            # Check if query mentions a specific stage
            stages = {}
            for d in ds.deals:
                stage = d.get("stage", "Unknown")
                stages[stage] = stages.get(stage, 0) + 1
            for stage in stages.keys():
                if stage.lower() in query.lower():
                    context_parts.append(f"- {stage} deals: {stages[stage]}")
    
    context = "\n\n".join(context_parts) if context_parts else f"Total available: {len(ds.companies)} companies, {len(ds.contacts)} contacts, {len(ds.deals)} deals, {len(ds.emails)} emails"
    
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
