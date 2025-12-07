import json

# Count sentiment values in emails
sentiments = {}
with open('output/emails.json', 'r') as f:
    emails = json.load(f)
    for email in emails:
        sentiment = email.get("sentiment", "Unknown")
        sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

print("Email sentiments:")
for sentiment, count in sorted(sentiments.items(), key=lambda x: x[1], reverse=True):
    print(f"  {sentiment}: {count}")
print(f"\nTotal emails: {len(emails)}")
