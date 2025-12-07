import os
os.environ['OPENAI_API_KEY'] = ''
os.environ['GROQ_API_KEY'] = ''

from app.agent import run_agent

edge_cases = [
    "What meetings do we have?",
    "Show me deals worth more than 100k",
    "Tell me about Nguyen-Allen",
    "List all people",
    "Something random that won't match"
]

print("=" * 80)
print("EDGE CASE TESTING - CONVERSATIONAL RESPONSES")
print("=" * 80)
print()

for query in edge_cases:
    print(f"Q: {query}")
    print("-" * 80)
    response = run_agent(query)
    print(f"A: {response}")
    print()
    print()
