# Demo Web App (RL Multi-hop Retrieval)

ChatGPT-style demo over the existing RL `ReasoningAgent`.
**FastAPI + MongoDB + React (Vite).** Chat storage uses MongoDB only (never Neo4j).

```
webapp/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, CORS, routers, startup
│   │   ├── config.py          # env settings
│   │   ├── db.py              # Motor (async MongoDB) + indexes
│   │   ├── security.py        # bcrypt + JWT + current-user dependency
│   │   ├── models.py          # Pydantic schemas
│   │   ├── agent_service.py   # >>> INTEGRATION POINT: ReasoningAgent.answer()
│   │   └── routers/
│   │       ├── auth.py        # /auth/signup, /auth/login
│   │       ├── threads.py     # /threads CRUD + messages
│   │       └── chat.py        # /chat
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # auth gate + thread/message state
│   │   ├── api.js             # fetch wrapper (JWT)
│   │   ├── components/Auth.jsx, Sidebar.jsx, Chat.jsx
│   │   ├── main.jsx, styles.css
│   ├── index.html, vite.config.js, package.json
├── MONGO_SCHEMA.md
└── README.md
```

## API design

| Method | Path | Auth | Body | Returns |
|---|---|---|---|---|
| POST | `/auth/signup` | – | `{email, username, password}` | `{access_token, user}` |
| POST | `/auth/login` | – | `{email, password}` | `{access_token, user}` |
| GET | `/threads` | ✅ | – | `[{id,title,created_at,updated_at}]` |
| POST | `/threads` | ✅ | `{title?}` | thread |
| PATCH | `/threads/{id}` | ✅ | `{title}` | thread |
| DELETE | `/threads/{id}` | ✅ | – | `{ok:true}` |
| GET | `/threads/{id}/messages` | ✅ | – | `[message]` |
| POST | `/chat` | ✅ | `{thread_id?, message}` | `{thread_id, answer, hops, confidence, reasoning_path, evidence, ...}` |
| GET | `/health` | – | – | `{status, agent_enabled}` |

Auth: send `Authorization: Bearer <access_token>`.

## Integration point

`backend/app/agent_service.py` → `run_inference(query)` calls:

```python
answer = agent.answer(query, deterministic=True)   # ReasoningAgent
# → answer.final_answer, answer.n_hops, answer.uncertainty,
#   answer.reasoning_path_text, answer.evidence_chain
```

mapped to `{answer, hops, confidence = 1 - uncertainty, reasoning_path, evidence}`.

The agent is built once via the project's `build_system(DEFAULT_CONFIG)` and loads the
trained checkpoint (`best`, else `imitation_final`) into `actor.policy`.

> If `AGENT_ENABLED=false` (default) or the build fails, `/chat` returns a clearly
> labelled **mock** answer so the entire UI + storage works without Neo4j / the model.

## Setup

### 1. MongoDB
Run a local MongoDB (or Docker):
```bash
docker run -d --name rlchat-mongo -p 27017:27017 mongo:7
```

### 2. Backend
```bash
cd webapp/backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env

# UI-only demo (no model): leave AGENT_ENABLED=false
# Real agent: set AGENT_ENABLED=true and
#   RESEARCH_PROJECT_PATH=E:/x/ecommerce-multihop-qa   (folder containing src/)
#   ...and make sure Neo4j is running + a checkpoint exists.

uvicorn app.main:app --reload --port 8000
```

### 3. Frontend
```bash
cd webapp/frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api → :8000)
```

## Run order
1. MongoDB up.
2. `uvicorn app.main:app --reload --port 8000`
3. `npm run dev` → open http://localhost:5173 → sign up → chat.

## Env variables (backend/.env)
| Var | Meaning |
|---|---|
| `MONGO_URI` | MongoDB connection string |
| `MONGO_DB` | database name (`rl_chat`) |
| `JWT_SECRET` | long random string |
| `JWT_EXPIRE_MIN` | token lifetime (minutes) |
| `CORS_ORIGINS` | `*` or comma-separated origins |
| `RESEARCH_PROJECT_PATH` | abs path to the RL project root (`src/` parent) |
| `AGENT_ENABLED` | `true` to load the real agent, else mock |
