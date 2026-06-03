# Adaptive Multi-hop Retrieval over E-commerce KG — TASK 2-3-4

Triển khai đầy đủ cơ chế **truy xuất đa bước thích nghi + RL + dừng theo độ bất định**
theo đúng đề cương `Research_Proposal_PrePhD.pdf`. Retrieval được mô hình hóa như một
**chuỗi quyết định tuần tự trên Knowledge Graph (MDP)** — KHÔNG phải vanilla RAG.

---

## 1. Cấu trúc & vai trò module

```
src/
├── config/settings.py          # Master config: Neo4j, encoder, RL, reward weights, stopping
│
├── kg/neo4j_connector.py       # KGConnector: get_node, get_neighbours (DYNAMIC), search_products
│
├── retrieval/                  # TASK 2 + TASK 3
│   ├── query_encoder.py        # q → vector + intent (constraints/aspects/numeric/negation)
│   ├── node_encoder.py         # 6 node types + 7 relation types → embeddings (LRU cache)
│   ├── state_builder.py        # s_t = [query|node|type|hop|history-GRU|evidence-GRU|uncertainty]
│   ├── semantic_scorer.py      # cosine/bilinear/MLP scoring + evidence saturation
│   ├── stopping.py             # AdaptiveStopping: MC Dropout + evidence spread + coverage
│   ├── adaptive_retriever.py   # Orchestrator inference → RetrievalResult
│   └── rl_retriever.py         # Bản RL-driven của vòng lặp truy xuất (policy điều hướng)
│
├── rl/                         # TASK 4
│   ├── kg_env.py               # KGEnvironment (Gym-style): reset/step/get_valid_actions; STOP = last
│   ├── policy_network.py       # StateEncoder + ActionEncoder + QueryGatedCrossAttention + actor/critic
│   ├── actor.py                # sampling, imitation_loss, evaluate_actions (PPO)
│   ├── critic.py               # GAE + clipped value loss
│   ├── reward.py               # 6-component multi-objective reward
│   ├── replay_buffer.py        # on-policy rollout, padded minibatch, GAE in-place
│   ├── ppo_trainer.py          # PPOTrainer (Stage 2) + ImitationTrainer (Stage 1)
│   ├── state_encoder.py        # LayerNorm + projection cho PPO batch
│   └── checkpoint.py           # save/load policy + optimizer + metadata
│
├── llm/llm_judge.py            # LLMJudge (reward shaping) + LLMSynthesiser (answer) — Claude API
│
├── reasoning/agent.py          # EvidenceAggregator + ExplainablePathBuilder + synthesis → AgentAnswer
│
└── evaluation/                 # metrics + evaluator: Hit@K, MRR, NDCG, path quality, efficiency

training_rl_agent.py            # Entry point: Stage1 → Stage2 pipeline (CLI)
```

---

## 2. Cách các module kết nối (luồng end-to-end)

### Inference (per query)
```
User query (English)
  → QueryEncoder           : q_vec + intent (aspects, numeric filters, negative constraints)
  → KGConnector.search     : anchor nodes (semantic similarity)
  → KGEnvironment.reset     : s_0 = (q, v_0, history=∅, evidence=∅)
  ┌─ LOOP ──────────────────────────────────────────────────────────┐
  │  StateBuilder           : s_t → state vector (dim 560)            │
  │  AdaptiveStopping        : U(s_t) < threshold ? → STOP            │
  │  KGEnv.get_valid_actions : dynamic action space từ neighbourhood  │
  │  PolicyNetwork (actor)   : π(a|s_t) → chọn action (move/edge/STOP)│
  │  KGEnv.step              : v_{t+1}, history+=, evidence+=         │
  └──────────────────────────────────────────────────────────────────┘
  → EvidenceAggregator     : C_T
  → LLMSynthesiser         : answer cuối
  → ExplainablePathBuilder : trajectory + evidence chain (giải thích được)
```

### Training
```
Stage 1 — ImitationTrainer : teacher = neighbour có cosine cao nhất với query
                             → warm-start policy (BC loss)
Stage 2 — PPOTrainer       : rollout trên KGEnvironment → RewardFunction (6 thành phần,
                             R_answer_quality lấy lazy từ LLMJudge ở terminal step)
                             → GAE (Critic) → clipped PPO update (Actor)
```

---

## 3. Cách chạy

```bash
# 0. Cài đặt
pip install -r requirements.txt

# 1. Biến môi trường
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export ANTHROPIC_API_KEY="sk-ant-..."        # cho LLMJudge / LLMSynthesiser

# 2. Smoke test kết nối + 1 episode
python training_rl_agent.py --test

# 3. Stage 1 — Imitation Learning (warm start)
python training_rl_agent.py --stage imitation --iterations 50

# 4. Stage 2 — PPO (resume từ checkpoint imitation)
python training_rl_agent.py --stage ppo --iterations 500 --resume outputs/checkpoints/imitation_final.pt

# 5. Chạy cả pipeline 2 giai đoạn liên tiếp
python training_rl_agent.py --stage all --iterations 500
```

---

## 4. Example usage (inference trực tiếp)

```python
from src.config.settings import DEFAULT_CONFIG
from src.kg.neo4j_connector import KGConnector
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.reasoning.agent import ReasoningAgent

kg = KGConnector(DEFAULT_CONFIG)
retriever = AdaptiveRetriever(DEFAULT_CONFIG, kg)          # load policy đã train
agent = ReasoningAgent(DEFAULT_CONFIG, kg)

query = "wireless earbuds with strong bass but not bulky"  # có negative constraint
result = retriever.retrieve(query)                          # RetrievalResult
answer = agent.answer(query, result)                        # AgentAnswer

print(answer.text)                 # câu trả lời cuối
print(answer.reasoning_path_text)  # đường đi suy luận giải thích được
for c in answer.evidence_chain:    # chuỗi bằng chứng có nguồn gốc
    print(c.source_node_id, c.content)
```

---

## 5. Logging / debug strategy

- Mỗi module dùng `logging.getLogger(__name__)`; cấu hình tập trung trong `settings.py`
  (`LOG_LEVEL`, format có timestamp + module).
- **Trajectory log**: `KGEnvironment` ghi mỗi step `(v_t, action, reward_breakdown, U_t)` →
  `outputs/logs/` để truy vết hành vi điều hướng.
- **Reward breakdown**: `RewardFunction` trả về dict tách 6 thành phần — bật `DEBUG` để xem
  từng thành phần đóng góp bao nhiêu (chẩn đoán over-/early-stopping).
- **Stopping diagnostics**: `AdaptiveStoppingModule` log `uncertainty`, `evidence_spread`,
  `policy_confidence`, `constraint_coverage` mỗi bước → biết model dừng vì lý do gì.
- **PPO metrics**: `ppo_trainer.py` log `policy_loss`, `value_loss`, `entropy`, `KL`,
  `clip_fraction`, `mean_episode_reward`, `mean_hops` mỗi iteration.
- **Checkpoints**: `CheckpointManager` lưu policy + optimizer + metadata vào
  `outputs/checkpoints/` theo iteration; dùng `--resume` để tiếp tục.

---

## 6. Bám sát methodology (checklist)

| Yêu cầu đề cương | Triển khai |
|---|---|
| Retrieval = sequential reasoning over KG | `kg_env.py` (MDP), `rl_retriever.py` |
| Dynamic action space | `KGEnv.get_valid_actions` từ neighbourhood runtime |
| KHÔNG BFS/DFS/fixed hop | `max_hops=10` chỉ là safety brake; logic core = policy + stopping |
| Adaptive stopping theo uncertainty | `stopping.py`: MC Dropout + evidence spread |
| Multi-objective reward | `reward.py`: 6 thành phần |
| 2-stage training (Imitation → PPO) | `ppo_trainer.py` |
| LLM chỉ weak-sup + synthesis | `llm_judge.py` (KHÔNG nằm trong retrieval loop) |
| Negative constraints | `query_encoder.py` intent + reward path quality |
| Explainability | `reasoning/agent.py`: trajectory + evidence chain |
| English query | `query_encoder.py` (sentence-transformer) |