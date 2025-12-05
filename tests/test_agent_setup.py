"""Simple test to verify agent setup and Groq connectivity."""

import os
from dotenv import load_dotenv

load_dotenv()

def test_data_loader():
    """Test data loader initialization."""
    from app.data_loader import DataStore
    
    store = DataStore(data_path="output")
    summary = store.get_summary()
    print("✓ DataStore loaded successfully")
    print(f"  - Companies: {summary['total_companies']}")
    print(f"  - Contacts: {summary['total_contacts']}")
    print(f"  - Deals: {summary['total_deals']}")
    return True


def test_groq_connection():
    """Test Groq LLM connection."""
    from langchain_groq import ChatGroq
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("✗ GROQ_API_KEY not set in .env")
        return False
    
    try:
        llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="mixtral-8x7b-32768"
        )
        print("✓ Groq LLM initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Groq connection failed: {e}")
        return False


def test_langsmith():
    """Test LangSmith setup."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    
    if api_key:
        print("✓ LangSmith API key configured")
        if tracing:
            print("  - Tracing: ENABLED")
        else:
            print("  - Tracing: disabled (set LANGSMITH_TRACING=true to enable)")
        return True
    else:
        print("✗ LANGSMITH_API_KEY not set (tracing disabled)")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ProspectIQ Agent Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    try:
        results.append(("DataStore", test_data_loader()))
    except Exception as e:
        print(f"✗ DataStore test failed: {e}")
        results.append(("DataStore", False))
    
    print()
    results.append(("Groq", test_groq_connection()))
    print()
    results.append(("LangSmith", test_langsmith()))
    
    print()
    print("=" * 60)
    if all(r[1] for r in results):
        print("All checks passed! ✓")
        print("You can now run: python -m app.chat")
    else:
        print("Some checks failed. See above for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
