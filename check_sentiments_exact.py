import json

# Check all unique sentiment values exactly as they appear
sentiments_exact = set()
sentiment_counts = {}

with open('output/emails.json', 'r') as f:
    emails = json.load(f)
    for email in emails:
        sentiment = email.get("sentiment", "Unknown")
        sentiments_exact.add(sentiment)
        
        # Count with exact case
        if sentiment not in sentiment_counts:
            sentiment_counts[sentiment] = 0
        sentiment_counts[sentiment] += 1

print("Exact sentiment values found:")
for sentiment in sorted(sentiments_exact):
    print(f"  '{sentiment}': {sentiment_counts[sentiment]}")

print(f"\nTotal: {sum(sentiment_counts.values())}")

# Also check case-insensitive positive
positive_any_case = sum(1 for e in emails if e.get("sentiment", "").lower() == "positive")
print(f"Case-insensitive 'positive' count: {positive_any_case}")
