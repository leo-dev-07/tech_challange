import os
os.environ['OPENAI_API_KEY'] = ''
os.environ['GROQ_API_KEY'] = ''

from app.agent import run_agent

test_queries = [
    "How many SaaS companies do we have?",
    "Tell me about Healthcare companies",
    "Who is Kimberly Wright?",
    "What deals are closing soon?",
    "Show me all contacts at Acme Corp",
    "Hello",
    "Hi there"
]

print("=" * 80)
print("CONVERSATIONAL AGENT RESPONSE TEST")
print("=" * 80)
print()

for query in test_queries:
    print(f"Q: {query}")
    print("-" * 80)
    response = run_agent(query)
    print(f"A: {response}")
    print()
    print()
