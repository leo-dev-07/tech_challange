"""Interactive chat CLI for the ProspectIQ query agent."""

import os
import sys
from dotenv import load_dotenv
from .agent import run_agent

# Load environment variables
load_dotenv()


def main():
    """Run an interactive chat session with the agent."""
    print("=" * 60)
    print("ProspectIQ Demo Data Query Agent")
    print("=" * 60)
    print("Chat with the agent to query sales data.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            print("\nAgent: ", end="", flush=True)
            response = run_agent(user_input)
            print(response)
            print()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()
