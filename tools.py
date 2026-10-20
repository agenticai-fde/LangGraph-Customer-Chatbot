"""Supabase-backed tools and memory helpers for the customer chatbot."""
from typing import Optional

from langchain_core.tools import tool

from config import get_db


def find_customer(email: str) -> Optional[dict]:
    rows = (get_db().table("customers").select("id,name,email,plan")
            .eq("email", email.strip().lower()).execute().data)
    return rows[0] if rows else None


def open_ticket(email: str, subject: str, body: str,
                order_number: Optional[str] = None, channel: str = "chat") -> Optional[int]:
    """Insert a support ticket; returns its id, or None if the email is unknown."""
    customer = find_customer(email)
    if not customer:
        return None
    row = get_db().table("support_tickets").insert({
        "customer_id": customer["id"],
        "order_number": (order_number or "").strip().upper() or None,
        "subject": subject.strip()[:200],
        "body": body.strip(),
        "channel": channel,
    }).execute().data[0]
    return row["id"]


# ---------- Tools the LLM can call ----------

@tool
def search_faqs(question: str) -> str:
    """Search the ShopEase help centre for policy answers: shipping, tracking, returns,
    refunds, warranty, payments, cancellations, address changes, plans, support hours."""
    rows = get_db().rpc("search_faqs", {"q": question, "k": 3}).execute().data
    if not rows:
        return "No matching FAQ found."
    return "\n\n".join(f"Q: {r['question']}\nA: {r['answer']}" for r in rows)


@tool
def get_order_status(order_number: str, email: str) -> str:
    """Get the status of one order. Needs the order number (e.g. ORD-1001) AND the
    email on the account, which is used to verify the customer."""
    rows = (get_db().table("orders")
            .select("order_number,status,order_date,quantity,total,products(name),customers!inner(email)")
            .eq("order_number", order_number.strip().upper())
            .eq("customers.email", email.strip().lower())
            .execute().data)
    if not rows:
        return f"No order {order_number} found for that email."
    o = rows[0]
    return (f"{o['order_number']}: {o['quantity']} x {o['products']['name']}, "
            f"ordered {o['order_date']}, total Rs {o['total']}, status: {o['status']}.")


@tool
def list_my_orders(email: str) -> str:
    """List every order for the customer with this email address."""
    customer = find_customer(email)
    if not customer:
        return "No customer account found for that email."
    rows = (get_db().table("orders")
            .select("order_number,status,order_date,total,products(name)")
            .eq("customer_id", customer["id"]).order("order_date", desc=True)
            .execute().data)
    if not rows:
        return f"{customer['name']} has no orders yet."
    return "\n".join(f"- {o['order_number']} ({o['order_date']}): {o['products']['name']}, "
                     f"Rs {o['total']}, {o['status']}" for o in rows)


@tool
def create_support_ticket(email: str, subject: str, details: str, order_number: str = "") -> str:
    """Open a support ticket for problems you cannot solve in chat (defective or damaged
    item, refund request, complaint). Only call after the customer agrees."""
    ticket_id = open_ticket(email, subject, details, order_number)
    if ticket_id is None:
        return "Could not open a ticket: no customer account found for that email."
    return f"Ticket #{ticket_id} created. Our support team will reply by email within 24 hours."


TOOLS = [search_faqs, get_order_status, list_my_orders, create_support_ticket]


# ---------- Conversation memory (persisted in Supabase) ----------

def load_history(session_id: str, limit: int) -> list[dict]:
    rows = (get_db().table("chat_messages").select("role,content")
            .eq("session_id", session_id).order("id", desc=True)  # id is strictly increasing; batch rows share created_at
            .limit(limit).execute().data)
    return list(reversed(rows))


def save_messages(session_id: str, messages: list[tuple[str, str]]) -> None:
    get_db().table("chat_messages").insert([
        {"session_id": session_id, "role": role, "content": content}
        for role, content in messages
    ]).execute()
