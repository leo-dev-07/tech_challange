"""
Demonstration of natural conversational responses from the ProspectIQ agent.
This script shows various query types and their conversational responses.
"""

import os
os.environ['OPENAI_API_KEY'] = ''
os.environ['GROQ_API_KEY'] = ''

from app.agent import run_agent

def demo_query(question: str, description: str = ""):
    """Run a query and display the result."""
    print(f"\n{'='*80}")
    print(f"QUERY: {question}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*80}")
    response = run_agent(question)
    print(f"\n{response}\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ProspectIQ CONVERSATIONAL RESPONSE DEMO")
    print("="*80)
    print("\nThis demonstrates how the agent provides human-like, contextual responses")
    print("that sound like a teacher explaining data to a student.\n")
    
    # Demonstrate greeting handling
    print("\n" + ">> GREETINGS")
    demo_query("Hello", "Generic greeting - friendly response without RAG data")
    demo_query("Hi there", "Greeting variation - properly detected")
    
    # Demonstrate company queries
    print("\n" + ">> COMPANY QUERIES")
    demo_query("Show me SaaS companies", "Multiple companies in single industry")
    demo_query("Tell me about Healthcare companies", "Multiple companies, different industry")
    demo_query("Jackson Ltd", "Search for specific company name")
    
    # Demonstrate contact queries  
    print("\n" + ">> CONTACT QUERIES")
    demo_query("Who is Kimberly Wright?", "Search for specific contact")
    
    # Demonstrate complex queries
    print("\n" + ">> COMPLEX QUERIES")
    demo_query("What deals are in progress?", "Deal-focused query")
    
    print("\n" + "="*80)
    print("Key Features Demonstrated:")
    print("  + Natural, varied opening sentences")
    print("  + Rich context (industry, employee count, averages)")
    print("  + Dynamic follow-up questions based on data type")
    print("  + Smart greeting detection")
    print("  + Human-like explanatory style")
    print("  + Proper formatting with bold names")
    print("="*80 + "\n")
