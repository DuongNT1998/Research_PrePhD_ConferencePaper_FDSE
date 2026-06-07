import datetime as dt
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..db import get_db
from ..models import ChatIn, ChatOut, MessageOut
from ..security import get_current_user
from ..agent_service import run_inference

router = APIRouter(tags=["chat"])


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


def _auto_title(text: str) -> str:
    words = text.strip().split()
    title = " ".join(words[:7])
    return (title[:60] + ("…" if len(title) > 60 else "")) or "New chat"


@router.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn, user=Depends(get_current_user)):
    db = get_db()
    now = dt.datetime.now(dt.timezone.utc)

    # Resolve or create the thread
    if body.thread_id:
        try:
            oid = ObjectId(body.thread_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Bad thread id")
        thread = await db.threads.find_one({"_id": oid, "user_id": user["_id"]})
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
    else:
        thread = {
            "user_id": user["_id"],
            "title": _auto_title(body.message),
            "created_at": now,
            "updated_at": now,
        }
        res = await db.threads.insert_one(thread)
        thread["_id"] = res.inserted_id

    # If this is the first message of an untitled thread, auto-title it.
    msg_count = await db.messages.count_documents({"thread_id": thread["_id"]})
    if msg_count == 0 and thread.get("title", "New chat") == "New chat":
        await db.threads.update_one(
            {"_id": thread["_id"]}, {"$set": {"title": _auto_title(body.message)}}
        )
        thread["title"] = _auto_title(body.message)

    # 1) store the user message
    user_msg = {
        "thread_id": thread["_id"],
        "user_id": user["_id"],
        "role": "user",
        "content": body.message,
        "meta": {},
        "created_at": now,
    }
    ures = await db.messages.insert_one(user_msg)
    user_msg["_id"] = ures.inserted_id

    # 2) run the RL agent (blocking → threadpool)
    result: Dict[str, Any] = await run_in_threadpool(run_inference, body.message)

    # 3) store the assistant message + metadata
    assistant_now = dt.datetime.now(dt.timezone.utc)
    assistant_meta = {
        "hops": result["hops"],
        "confidence": result["confidence"],
        "reasoning_path": result["reasoning_path"],
        "evidence": result["evidence"],
    }
    assistant_msg = {
        "thread_id": thread["_id"],
        "user_id": user["_id"],
        "role": "assistant",
        "content": result["answer"],
        "meta": assistant_meta,
        "created_at": assistant_now,
    }
    ares = await db.messages.insert_one(assistant_msg)
    assistant_msg["_id"] = ares.inserted_id

    # touch thread updated_at
    await db.threads.update_one(
        {"_id": thread["_id"]}, {"$set": {"updated_at": assistant_now}}
    )

    return ChatOut(
        thread_id=str(thread["_id"]),
        answer=result["answer"],
        hops=result["hops"],
        confidence=result["confidence"],
        reasoning_path=result["reasoning_path"],
        evidence=result["evidence"],
        thread_title=thread["title"],
        user_message=MessageOut(
            id=str(user_msg["_id"]), role="user", content=user_msg["content"],
            meta={}, created_at=_iso(now),
        ),
        assistant_message=MessageOut(
            id=str(assistant_msg["_id"]), role="assistant",
            content=assistant_msg["content"], meta=assistant_meta,
            created_at=_iso(assistant_now),
        ),
    )