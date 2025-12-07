"""Test the feedback system with simulated interactions."""

import os
os.environ['OPENAI_API_KEY'] = ''
os.environ['GROQ_API_KEY'] = ''

from app.agent import run_agent
from app.feedback import FeedbackStore

# Initialize feedback store
feedback = FeedbackStore()

print("=" * 80)
print("FEEDBACK SYSTEM TEST")
print("=" * 80)
print()

# Query 1: SaaS companies
print("1. User asks about SaaS companies")
print("-" * 80)
query1 = "Show me SaaS companies"
response1 = run_agent(query1)
print(f"Response:\n{response1}\n")

# Simulate user marking as good
print("User marks as :good")
feedback.add_feedback(query1, response1, "good")
print()

# Query 2: Healthcare companies
print("2. User asks about Healthcare companies")
print("-" * 80)
query2 = "Tell me about Healthcare companies"
response2 = run_agent(query2)
print(f"Response:\n{response2}\n")

# Simulate user marking as bad with correction
print("User marks as :bad and provides correction")
correction = "There are 6 healthcare companies: Burnett-Stafford, Short-Rivers, Reeves LLC, Jackson Ltd, Hardin and Sons, and Perez-Kemp. They're all in the healthcare sector with sizes ranging from 50 to 2000 employees."
feedback.add_feedback(query2, response2, "bad", correction=correction, tags=["too_verbose"])
print()

# Query 3: Contact search
print("3. User asks who Kimberly Wright is")
print("-" * 80)
query3 = "Who is Kimberly Wright?"
response3 = run_agent(query3)
print(f"Response:\n{response3}\n")

# Mark as bad with tags
print("User marks as :bad with tags")
feedback.add_feedback(query3, response3, "bad", tags=["needs_company_info", "missing_email"])
print()

# Show statistics
print("=" * 80)
print("FEEDBACK STATISTICS")
print("=" * 80)
stats = feedback.get_feedback_stats()
print(f"Total interactions tracked: {stats['total_interactions']}")
print(f"Ratings: {stats['ratings']}")
print(f"Common issues reported: {dict(feedback.get_common_issues(5))}")
print()

# Show recent feedback
print("=" * 80)
print("RECENT FEEDBACK")
print("=" * 80)
recent = feedback.get_recent_feedback(3)
for i, entry in enumerate(recent, 1):
    print(f"\n{i}. Query: {entry['query']}")
    print(f"   Rating: {entry['rating']}")
    if entry.get('tags'):
        print(f"   Tags: {', '.join(entry['tags'])}")
    if entry.get('correction'):
        print(f"   Correction provided: Yes")

print("\n" + "=" * 80)
print("Now testing if agent avoids problematic patterns...")
print("=" * 80)
print()

# Test if agent uses corrections
print("Testing same query again (should potentially use correction):")
print(f"Query: {query2}")
should_avoid = feedback.should_avoid_pattern(query2)
print(f"Should avoid this pattern: {should_avoid}")
if should_avoid:
    corrections = feedback.get_corrections_for_pattern(query2)
    if corrections:
        print(f"Found correction to use:\n{corrections[0]}")

print("\n" + "=" * 80)
