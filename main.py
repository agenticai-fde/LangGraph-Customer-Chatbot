"""Chat with the ShopEase customer assistant in the terminal.

    python main.py                  start a new conversation
    python main.py --session 1a2b   resume a saved conversation
"""
import argparse
import uuid

from graph import graph


def main() -> None:
    parser = argparse.ArgumentParser(description="ShopEase customer chatbot")
    parser.add_argument("--session", help="session id to resume")
    args = parser.parse_args()

    session_id = args.session or uuid.uuid4().hex[:8]
    print(f"ShopEase assistant - session {session_id}. Type 'exit' to quit.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        try:
            result = graph.invoke({"session_id": session_id, "user_input": text})
            print(f"Ava: {result['reply']}\n")
        except Exception as e:  # keep the chat alive on LLM / network errors
            print(f"[error] {e}\n")

    print(f"\nBye! Resume later with: python main.py --session {session_id}")


if __name__ == "__main__":
    main()
