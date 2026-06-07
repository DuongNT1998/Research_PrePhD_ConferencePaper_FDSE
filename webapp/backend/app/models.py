from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field


# ---- Auth ----
class SignupIn(BaseModel):
    email: EmailStr
    username: str = Field(min_length=2, max_length=40)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: EmailStr
    username: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---- Threads ----
class ThreadOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class CreateThreadIn(BaseModel):
    title: Optional[str] = None


class RenameThreadIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)


# ---- Messages / Chat ----
class MessageOut(BaseModel):
    id: str
    role: str               # "user" | "assistant"
    content: str
    meta: Dict[str, Any] = {}
    created_at: str


class ChatIn(BaseModel):
    thread_id: Optional[str] = None    # if None, a new thread is created
    message: str = Field(min_length=1)


class ChatOut(BaseModel):
    thread_id: str
    answer: str
    hops: int
    confidence: float
    reasoning_path: List[str] = []
    evidence: List[str] = []
    user_message: MessageOut
    assistant_message: MessageOut
    thread_title: str