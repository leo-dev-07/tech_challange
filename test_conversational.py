import os
os.environ['OPENAI_API_KEY'] = ''  # Disable OpenAI
os.environ['GROQ_API_KEY'] = ''    # Disable Groq

from app.agent import _call_llm_with_context, _retrieve_rag_context, _format_snippets_as_response

# Test the functions directly
user_query = 'How many SaaS companies do we have?'
snippets = _retrieve_rag_context(user_query)
print('Snippets retrieved:')
print(f'  Companies: {len(snippets.get("companies", []))}')
print(f'  Contacts: {len(snippets.get("contacts", []))}')
print(f'  Deals: {len(snippets.get("deals", []))}')
print()

print("=" * 70)
print("Testing _format_snippets_as_response directly:")
print("=" * 70)
formatted = _format_snippets_as_response(snippets)
print(formatted)
print()

print("=" * 70)
print("Testing _call_llm_with_context (should use formatted since no LLM):")
print("=" * 70)
response = _call_llm_with_context(user_query, snippets)
print(response)
