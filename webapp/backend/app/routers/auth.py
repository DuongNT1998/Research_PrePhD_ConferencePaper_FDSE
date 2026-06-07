import datetime as dt

from bson import ObjectId
from fastapi import APIRouter, HTTPException

from ..db import get_db
from ..models import SignupIn, LoginIn, TokenOut, UserOut
from ..security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_out(user: dict) -> UserOut:
    return UserOut(id=str(user["_id"]), email=user["email"], username=user["username"])


@router.post("/signup", response_model=TokenOut)
async def signup(body: SignupIn):
    db = get_db()
    existing = await db.users.find_one({"email": body.email.lower()})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    doc = {
        "email": body.email.lower(),
        "username": body.username,
        "password_hash": hash_password(body.password),
        "created_at": dt.datetime.now(dt.timezone.utc),
    }
    res = await db.users.insert_one(doc)
    doc["_id"] = res.inserted_id
    token = create_access_token(str(res.inserted_id))
    return TokenOut(access_token=token, user=_user_out(doc))


@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn):
    db = get_db()
    user = await db.users.find_one({"email": body.email.lower()})
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(str(user["_id"]))
    return TokenOut(access_token=token, user=_user_out(user))