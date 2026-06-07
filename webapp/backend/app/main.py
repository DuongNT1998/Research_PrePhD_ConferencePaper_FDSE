import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import connect_db, close_db
from .routers import auth, threads, chat
from .agent_service import agent_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

settings = get_settings()
app = FastAPI(title="RL Multi-hop Retrieval — Demo API", version="1.0.0")

origins = ["*"] if settings.cors_origins.strip() == "*" else [
    o.strip() for o in settings.cors_origins.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(threads.router)
app.include_router(chat.router)


@app.on_event("startup")
async def _startup():
    await connect_db()


@app.on_event("shutdown")
async def _shutdown():
    await close_db()


@app.get("/health")
async def health():
    return {"status": "ok", "agent_enabled": settings.agent_enabled}


@app.get("/agent/status")
async def agent_status_route():
    return agent_status()