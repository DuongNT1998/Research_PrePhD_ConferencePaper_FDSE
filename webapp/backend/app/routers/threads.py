import datetime as dt
from typing import Any, Dict, List

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from ..db import get_db
from ..models import ThreadOut, CreateThreadIn, RenameThreadIn, MessageOut
from ..security import get_current_user

router = APIRouter(prefix="/threads", tags=["threads"])


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


def _thread_out(t: Dict[str, Any]) -> ThreadOut:
    return ThreadOut(
        id=str(t["_id"]),
        title=t.get("title", "New chat"),
        created_at=_iso(t.get("created_at")),
        updated_at=_iso(t.get("updated_at")),
    )


def _msg_out(m: Dict[str, Any]) -> MessageOut:
    return MessageOut(
        id=str(m["_id"]),
        role=m["role"],
        content=m["content"],
        meta=m.get("meta", {}),
        created_at=_iso(m.get("created_at")),
    )


@router.get("", response_model=List[ThreadOut])
async def list_threads(user=Depends(get_current_user)):
    db = get_db()
    cur = db.threads.find({"user_id": user["_id"]}).sort("updated_at", -1)
    return [_thread_out(t) async for t in cur]


@router.post("", response_model=ThreadOut)
async def create_thread(body: CreateThreadIn, user=Depends(get_current_user)):
    db = get_db()
    now = dt.datetime.now(dt.timezone.utc)
    doc = {
        "user_id": user["_id"],
        "title": (body.title or "New chat").strip()[:120],
        "created_at": now,
        "updated_at": now,
    }
    res = await db.threads.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _thread_out(doc)


async def _owned_thread(db, thread_id: str, user) -> Dict[str, Any]:
    try:
        oid = ObjectId(thread_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Bad thread id")
    t = await db.threads.find_one({"_id": oid, "user_id": user["_id"]})
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")
    return t


@router.patch("/{thread_id}", response_model=ThreadOut)
async def rename_thread(thread_id: str, body: RenameThreadIn, user=Depends(get_current_user)):
    db = get_db()
    t = await _owned_thread(db, thread_id, user)
    await db.threads.update_one(
        {"_id": t["_id"]},
        {"$set": {"title": body.title.strip()[:120],
                  "updated_at": dt.datetime.now(dt.timezone.utc)}},
    )
    t = await db.threads.find_one({"_id": t["_id"]})
    return _thread_out(t)


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str, user=Depends(get_current_user)):
    db = get_db()
    t = await _owned_thread(db, thread_id, user)
    await db.messages.delete_many({"thread_id": t["_id"]})
    await db.threads.delete_one({"_id": t["_id"]})
    return {"ok": True, "deleted": thread_id}


@router.get("/{thread_id}/messages", response_model=List[MessageOut])
async def get_messages(thread_id: str, user=Depends(get_current_user)):
    db = get_db()
    t = await _owned_thread(db, thread_id, user)
    cur = db.messages.find({"thread_id": t["_id"]}).sort("created_at", 1)
    return [_msg_out(m) async for m in cur]