"""FastAPI service for the customer chatbot.

    uvicorn api:app --port 8001      then open http://127.0.0.1:8001/docs
"""
import os
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from ollama import ResponseError
from pydantic import BaseModel, Field

from graph import HISTORY_MESSAGES, graph
from tools import load_history

API_KEY = os.getenv("API_KEY")  # when set, every request needs header X-API-Key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


app = FastAPI(title="Customer Chatbot",
              description="ShopEase customer assistant with FAQ search, verified order lookup, "
                          "persistent memory and human handoff.",
              dependencies=[Depends(require_api_key)])


@app.exception_handler(ResponseError)
def llm_unavailable(request, exc: ResponseError):
    return JSONResponse(status_code=503, content={"detail": f"LLM unavailable: {exc.error}"})


class ChatRequest(BaseModel):
    message: str = Field(description="Customer message, e.g. 'Where is my order ORD-1009?'")
    session_id: Optional[str] = Field(None, description="Omit to start a new conversation")


class ChatResponse(BaseModel):
    session_id: str
    route: str = Field(description="agent, handoff or blocked")
    reply: str


@app.post("/chat", response_model=ChatResponse, operation_id="chat")
def chat(data: ChatRequest):
    session_id = data.session_id or uuid.uuid4().hex[:8]
    result = graph.invoke({"session_id": session_id, "user_input": data.message})
    return ChatResponse(session_id=session_id, route=result["route"], reply=result["reply"])


@app.get("/sessions/{session_id}/history", operation_id="getHistory")
def history(session_id: str, limit: int = HISTORY_MESSAGES):
    return {"session_id": session_id, "messages": load_history(session_id, limit)}


@app.get("/health")
def health():
    return {"status": "ok"}
