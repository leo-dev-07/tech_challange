"""Interactive chat CLI for the ProspectIQ query agent."""

import os
import sys
from dotenv import load_dotenv
from .agent import run_agent
from .feedback import FeedbackStore

# Load environment variables
load_dotenv()

# Initialize feedback store
feedback_store = FeedbackStore()


def show_help():
    """Show available commands."""
    print("\n" + "=" * 60)
    print("AVAILABLE COMMANDS")
    print("=" * 60)
    print("After each response, you can provide feedback:")
    print("  :good          - Mark response as helpful")
    print("  :bad           - Mark response as unhelpful")
    print("  :correct TEXT  - Provide correction/better response")
    print("  :tags TAG1,TAG2 - Add tags (e.g., :tags too_long,wrong_data)")
    print("  :stats         - Show feedback statistics")
    print("  :recent        - Show recent feedback")
    print("  :help          - Show this help message")
    print("  :exit          - Quit the chat")
    print("=" * 60 + "\n")


def process_feedback_command(command: str, last_query: str, last_response: str) -> bool:
    """Process feedback commands. Returns True if it was a feedback command."""
    if not command.startswith(":"):
        return False
    
    parts = command.split(" ", 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    
    if cmd == ":good":
        feedback_store.add_feedback(last_query, last_response, "good")
        print("Thanks! Feedback recorded.")
        return True
    
    elif cmd == ":bad":
        feedback_store.add_feedback(last_query, last_response, "bad")
        print("Thanks for the feedback. I'll try to improve.")
        return True
    
    elif cmd == ":correct":
        if arg:
            feedback_store.add_feedback(last_query, last_response, "bad", correction=arg)
            print(f"Correction recorded. I'll learn from this.")
        else:
            print("Usage: :correct TEXT (provide the better response)")
        return True
    
    elif cmd == ":tags":
        if arg:
            tags = [t.strip() for t in arg.split(",")]
            feedback_store.add_feedback(last_query, last_response, "bad", tags=tags)
            print(f"Tags recorded: {', '.join(tags)}")
        else:
            print("Usage: :tags TAG1,TAG2,TAG3")
        return True
    
    elif cmd == ":stats":
        stats = feedback_store.get_feedback_stats()
        print("\n" + "=" * 60)
        print("FEEDBACK STATISTICS")
        print("=" * 60)
        print(f"Total interactions: {stats['total_interactions']}")
        print(f"Ratings: {stats['ratings']}")
        if stats['tags']:
            print(f"Common issues: {dict(feedback_store.get_common_issues(5))}")
        print("=" * 60 + "\n")
        return True
    
    elif cmd == ":recent":
        recent = feedback_store.get_recent_feedback(5)
        if recent:
            print("\n" + "=" * 60)
            print("RECENT FEEDBACK")
            print("=" * 60)
            for i, entry in enumerate(recent, 1):
                print(f"\n{i}. Query: {entry['query'][:50]}...")
                print(f"   Rating: {entry['rating']}")
                if entry.get('tags'):
                    print(f"   Tags: {', '.join(entry['tags'])}")
            print("\n" + "=" * 60 + "\n")
        else:
            print("No feedback recorded yet.")
        return True
    
    elif cmd == ":help":
        show_help()
        return True
    
    elif cmd == ":exit":
        return "exit"
    
    return False


def main():
    """Run an interactive chat session with the agent."""
    print("=" * 60)
    print("ProspectIQ Demo Data Query Agent")
    print("=" * 60)
    print("Chat with the agent to query sales data.")
    print("Type ':help' for feedback commands or 'exit' to quit.\n")

    last_query = ""
    last_response = ""

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            # Check for feedback commands
            feedback_result = process_feedback_command(user_input, last_query, last_response)
            if feedback_result == "exit":
                print("Goodbye!")
                break
            elif feedback_result:
                # It was a feedback command
                continue
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            print("\nAgent: ", end="", flush=True)
            response = run_agent(user_input)
            print(response)
            
            # Store for feedback
            last_query = user_input
            last_response = response
            
            print("\n(Type ':good', ':bad', ':correct TEXT', or ':help' for feedback)\n")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()
