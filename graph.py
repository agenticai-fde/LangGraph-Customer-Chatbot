"""Customer chatbot graph.

load_memory -> guardrail -> agent    -> save_memory
                         -> handoff  -> save_memory
                         -> blocked  -> save_memory
"""
import re
from datetime import date
from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from config import get_llm
from tools import TOOLS, load_history, open_ticket, save_messages

HISTORY_MESSAGES = 20
MAX_INPUT_CHARS = 2000
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
ORDER_RE = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
HANDOFF_RE = re.compile(r"\b(human|real person|live agent|representative|"
                        r"(speak|talk) to (someone|a person|a manager))\b", re.I)
INJECTION_RE = re.compile(r"ignore (all |your |the )?(previous|above|prior) (instructions|rules)|system prompt|you are now", re.I)
HANDOFF_ASK = "I can connect you with a human agent."

SYSTEM_PROMPT = f"""You are Ava, the customer assistant for ShopEase, an online electronics store.
Today is {date.today():%d %b %Y}.

Rules:
- Answer policy questions (shipping, tracking, returns, refunds, warranty, payments, plans,
  support hours) only from search_faqs results. Do not invent policies.
- To check an order you need BOTH the order number (like ORD-1001) and the account email.
  Ask for whatever is missing; never guess either one.
- For problems you cannot solve in chat (defective item, refund, complaint), offer to open a
  support ticket and call create_support_ticket only after the customer agrees.
- Keep replies short (2-4 sentences), friendly, plain text.
- For anything unrelated to ShopEase, say you can only help with ShopEase orders and policies."""

agent = create_agent(get_llm(), TOOLS, system_prompt=SYSTEM_PROMPT)


class ChatState(TypedDict, total=False):
    session_id: str
    user_input: str
    history: list[dict]
    route: str
    reply: str


def load_memory(state: ChatState) -> dict:
    return {"history": load_history(state["session_id"], HISTORY_MESSAGES)}


def guardrail(state: ChatState) -> dict:
    text = state["user_input"].strip()
    history = state.get("history", [])
    waiting_for_email = bool(history) and history[-1]["content"].startswith(HANDOFF_ASK)

    if not text:
        return {"route": "blocked", "reply": "Please type your question."}
    if len(text) > MAX_INPUT_CHARS:
        return {"route": "blocked",
                "reply": f"That message is too long. Please keep it under {MAX_INPUT_CHARS} characters."}
    if INJECTION_RE.search(text):
        return {"route": "blocked",
                "reply": "I can only help with ShopEase orders, products and policies."}
    if HANDOFF_RE.search(text) or (waiting_for_email and EMAIL_RE.search(text)):
        return {"route": "handoff"}
    return {"route": "agent"}


def run_agent(state: ChatState) -> dict:
    messages = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
                for m in state.get("history", [])]
    messages.append(HumanMessage(state["user_input"]))
    result = agent.invoke({"messages": messages})
    return {"reply": result["messages"][-1].content}


def handoff(state: ChatState) -> dict:
    """Escalate to a human by opening a ticket with the conversation transcript."""
    history = state.get("history", [])
    conversation = history + [{"role": "user", "content": state["user_input"]}]
    all_text = " ".join(m["content"] for m in conversation)
    email = next(iter(EMAIL_RE.findall(all_text)), None)
    if not email:
        return {"reply": f"{HANDOFF_ASK} Please share the email address on your account "
                         "so I can open a ticket for the support team."}

    order = next(iter(ORDER_RE.findall(all_text)), None)
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in conversation[-8:])
    ticket_id = open_ticket(email, "Chat escalation: customer asked for a human",
                            f"Escalated from chat session {state['session_id']}.\n\n{transcript}",
                            order_number=order)
    if ticket_id is None:
        return {"reply": f"I couldn't find an account for {email}. Please check the email address."}
    return {"reply": f"Done. I've opened ticket #{ticket_id} with our support team and shared "
                     "this conversation. A human agent will email you within 24 hours."}


def save_memory(state: ChatState) -> dict:
    save_messages(state["session_id"],
                  [("user", state["user_input"]), ("assistant", state["reply"])])
    return {}


builder = StateGraph(ChatState)
builder.add_node("load_memory", load_memory)
builder.add_node("guardrail", guardrail)
builder.add_node("agent", run_agent)
builder.add_node("handoff", handoff)
builder.add_node("save_memory", save_memory)

builder.add_edge(START, "load_memory")
builder.add_edge("load_memory", "guardrail")
builder.add_conditional_edges("guardrail", lambda s: s["route"],
                              {"agent": "agent", "handoff": "handoff", "blocked": "save_memory"})
builder.add_edge("agent", "save_memory")
builder.add_edge("handoff", "save_memory")
builder.add_edge("save_memory", END)

graph = builder.compile()
