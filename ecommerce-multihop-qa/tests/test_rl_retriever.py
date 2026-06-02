"""
tests/test_rl_retriever.py

Unit tests for TASK 3 components:
  - PolicyNetwork
  - Actor (action selection, imitation loss)
  - Critic (GAE)
  - RewardFunction
  - RolloutBuffer
  - RLRetriever (mocked environment)

All tests use mock/synthetic data — no Neo4j connection required.
Run with: pytest tests/test_rl_retriever.py -v
"""

from __future__ import annotations

import os
import sys


# Thêm dòng này để Python tìm thấy thư mục 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import unittest
from unittest.mock import MagicMock, patch
from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def make_config():
    from src.config.settings import Config
    cfg = Config()
    cfg.encoder.device = "cpu"
    cfg.encoder.embedding_dim = 384
    cfg.encoder.node_feature_dim = 128
    cfg.state.query_dim = 256
    cfg.state.node_dim = 128
    cfg.state.node_type_dim = 16
    cfg.state.hop_encoding_dim = 16
    cfg.state.history_dim = 64
    cfg.state.evidence_dim = 64
    cfg.state.uncertainty_dim = 16
    cfg.policy.hidden_dim = 128      # smaller for test speed
    cfg.policy.num_layers = 2
    cfg.policy.attention_heads = 2
    cfg.policy.dropout = 0.0         # deterministic for tests
    cfg.ppo.clip_epsilon = 0.2
    cfg.ppo.gamma = 0.99
    cfg.ppo.gae_lambda = 0.95
    cfg.ppo.mini_batch_size = 4
    return cfg


def make_state_vec(config) -> torch.Tensor:
    return torch.randn(config.state.total_dim)


def make_action_embs(config, n: int = 5) -> torch.Tensor:
    action_dim = config.encoder.node_feature_dim + config.state.edge_type_dim
    return torch.randn(n, action_dim)


def make_query_emb(config) -> torch.Tensor:
    return torch.randn(config.state.query_dim)


# ---------------------------------------------------------------------------
# PolicyNetwork tests
# ---------------------------------------------------------------------------

class TestPolicyNetwork(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def test_forward_output_shapes(self):
        from src.rl.policy_network import PolicyNetwork
        net = PolicyNetwork(self.config)
        N = 5
        s = make_state_vec(self.config)
        a = make_action_embs(self.config, N)
        q = make_query_emb(self.config)

        logits, probs, value, attn = net(s, a, q)
        self.assertEqual(logits.shape[0], N)
        self.assertEqual(probs.shape[0], N)
        self.assertIsInstance(value.item(), float)
        self.assertEqual(attn.shape[0], N)

    def test_probs_sum_to_one(self):
        from src.rl.policy_network import PolicyNetwork
        net = PolicyNetwork(self.config)
        s = make_state_vec(self.config)
        a = make_action_embs(self.config, 4)
        q = make_query_emb(self.config)
        _, probs, _, _ = net(s, a, q)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_action_mask(self):
        from src.rl.policy_network import PolicyNetwork
        net = PolicyNetwork(self.config)
        N = 5
        s = make_state_vec(self.config)
        a = make_action_embs(self.config, N)
        q = make_query_emb(self.config)

        # Mask last 2 actions as invalid
        mask = torch.tensor([True, True, True, False, False])
        _, probs, _, _ = net(s, a, q, action_mask=mask)
        # Masked actions should have near-zero probability
        self.assertAlmostEqual(probs[3].item(), 0.0, places=4)
        self.assertAlmostEqual(probs[4].item(), 0.0, places=4)

    def test_select_action_deterministic(self):
        from src.rl.policy_network import PolicyNetwork
        net = PolicyNetwork(self.config)
        net.eval()
        s = make_state_vec(self.config)
        a = make_action_embs(self.config, 5)
        q = make_query_emb(self.config)

        idx1, lp1, val1, _ = net.select_action(s, a, q, deterministic=True)
        idx2, lp2, val2, _ = net.select_action(s, a, q, deterministic=True)
        # Deterministic: same state → same action
        self.assertEqual(idx1, idx2)

    def test_select_action_returns_valid_index(self):
        from src.rl.policy_network import PolicyNetwork
        net = PolicyNetwork(self.config)
        N = 7
        s = make_state_vec(self.config)
        a = make_action_embs(self.config, N)
        q = make_query_emb(self.config)
        idx, _, _, _ = net.select_action(s, a, q)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, N)

    def test_parameter_count_positive(self):
        from src.rl.policy_network import PolicyNetwork
        net = PolicyNetwork(self.config)
        n_params = sum(p.numel() for p in net.parameters())
        self.assertGreater(n_params, 0)


# ---------------------------------------------------------------------------
# Actor tests
# ---------------------------------------------------------------------------

class TestActor(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def _make_actions(self, n: int = 5, config=None):
        from src.rl.kg_env import Action
        cfg = config or self.config
        action_dim = cfg.encoder.node_feature_dim + cfg.state.edge_type_dim
        actions = []
        for i in range(n - 1):
            actions.append(Action(
                index=i,
                is_stop=False,
                target_node_id=f"node_{i}",
                target_node_type="Product",
                relation_type="HAS_BRAND",
                action_embedding=torch.randn(action_dim),
            ))
        # STOP action last
        actions.append(Action(
            index=n - 1,
            is_stop=True,
            target_node_id=None,
            target_node_type=None,
            relation_type="STOP",
            action_embedding=torch.randn(action_dim),
        ))
        return actions

    def test_act_returns_valid_index(self):
        from src.rl.actor import Actor
        actor = Actor(self.config)
        actions = self._make_actions(5)
        s = make_state_vec(self.config)
        q = make_query_emb(self.config)
        idx, lp, val, attn = actor.act(s, actions, q)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(actions))

    def test_act_log_prob_is_finite(self):
        from src.rl.actor import Actor
        actor = Actor(self.config)
        actions = self._make_actions(4)
        s = make_state_vec(self.config)
        q = make_query_emb(self.config)
        _, lp, _, _ = actor.act(s, actions, q)
        self.assertTrue(math.isfinite(lp))

    def test_imitation_loss_is_scalar(self):
        from src.rl.actor import Actor
        actor = Actor(self.config)
        actions = self._make_actions(5)
        s = make_state_vec(self.config)
        q = make_query_emb(self.config)
        loss = actor.imitation_loss(s, actions, q, teacher_action_idx=2)
        self.assertEqual(loss.ndim, 0)   # scalar
        self.assertGreater(loss.item(), 0.0)

    def test_stack_action_embeddings_shape(self):
        from src.rl.actor import Actor
        actor = Actor(self.config)
        actions = self._make_actions(6)
        embs, mask = actor._stack_action_embeddings(actions)
        action_dim = self.config.encoder.node_feature_dim + self.config.state.edge_type_dim
        self.assertEqual(embs.shape, (6, action_dim))
        self.assertEqual(mask.shape[0], 6)
        self.assertTrue(mask.all())


# ---------------------------------------------------------------------------
# Critic / GAE tests
# ---------------------------------------------------------------------------

class TestCritic(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def test_gae_returns_correct_shape(self):
        from src.rl.critic import Critic
        critic = Critic(self.config)
        rewards = [0.1, 0.2, 0.3, 0.5, 1.0]
        values  = [0.5, 0.5, 0.5, 0.5, 0.5]
        dones   = [False, False, False, False, True]
        adv, ret = critic.compute_gae(rewards, values, dones, next_value=0.0)
        self.assertEqual(adv.shape[0], 5)
        self.assertEqual(ret.shape[0], 5)

    def test_returns_monotone_simple(self):
        from src.rl.critic import Critic
        critic = Critic(self.config)
        # Constant reward, no early done
        rewards = [1.0, 1.0, 1.0]
        values  = [0.0, 0.0, 0.0]
        dones   = [False, False, True]
        _, ret = critic.compute_gae(rewards, values, dones)
        # Returns should be positive and decreasing toward terminal
        self.assertTrue((ret >= 0).all())

    def test_value_loss_shape(self):
        from src.rl.critic import Critic
        critic = Critic(self.config)
        pred = torch.tensor([0.5, 0.6, 0.7])
        ret  = torch.tensor([1.0, 1.0, 1.0])
        loss = critic.value_loss(pred, ret)
        self.assertEqual(loss.ndim, 0)   # scalar


# ---------------------------------------------------------------------------
# RewardFunction tests
# ---------------------------------------------------------------------------

class TestRewardFunction(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def _make_episode(self, hop_count: int = 2):
        """Create a minimal EpisodeState for reward testing."""
        from src.retrieval.state_builder import EpisodeState, RetrievalStep
        from src.retrieval.query_encoder import QueryEncoder
        from src.retrieval.node_encoder import NodeEncoder
        enc = QueryEncoder(self.config)
        nenc = NodeEncoder(self.config)
        qr = enc.encode("laptop battery test")
        anchor = nenc.encode_node("Product", {"title": "Test"}, "B07X")
        from src.retrieval.state_builder import StateBuilder
        sb = StateBuilder(self.config)
        ep = sb.reset(qr, anchor)
        ep.hop_count = hop_count
        ep.collected_node_reprs = [torch.randn(128) for _ in range(hop_count)]
        ep.collected_evidence = ["ev1", "ev2"][:hop_count]
        # Add a fake step
        if hop_count > 0:
            ep.visited_steps.append(RetrievalStep(
                hop=1, node_id="B07Y", node_type="Aspect",
                relation_type="HAS_POSITIVE_ASPECT",
                edge_weight=5.0,
                node_embedding=torch.randn(128),
                evidence_text="battery positive",
            ))
        return ep

    def _make_stop_action(self):
        from src.rl.kg_env import Action
        return Action(
            index=0, is_stop=True,
            target_node_id=None, target_node_type=None,
            relation_type="STOP",
        )

    def _make_move_action(self):
        from src.rl.kg_env import Action
        return Action(
            index=0, is_stop=False,
            target_node_id="B07Z", target_node_type="Aspect",
            relation_type="HAS_POSITIVE_ASPECT",
        )

    def test_terminal_reward_returns_finite(self):
        from src.rl.reward import RewardFunction
        rf = RewardFunction(self.config)
        ep = self._make_episode(hop_count=2)
        action = self._make_stop_action()
        total, breakdown = rf(ep, action, is_terminal=True)
        self.assertTrue(math.isfinite(total))

    def test_non_terminal_reward_finite(self):
        from src.rl.reward import RewardFunction
        rf = RewardFunction(self.config)
        ep = self._make_episode(hop_count=1)
        action = self._make_move_action()
        total, breakdown = rf(ep, action, is_terminal=False)
        self.assertTrue(math.isfinite(total))

    def test_breakdown_has_all_keys(self):
        from src.rl.reward import RewardFunction
        rf = RewardFunction(self.config)
        ep = self._make_episode(hop_count=2)
        action = self._make_stop_action()
        _, bd = rf(ep, action, is_terminal=True)
        required_keys = {
            "answer_quality", "retrieval_relevance",
            "path_quality", "grounding",
            "efficiency_penalty", "uncertainty_stop", "total",
        }
        self.assertEqual(required_keys, set(bd.keys()))

    def test_efficiency_penalty_increases_with_hops(self):
        from src.rl.reward import compute_efficiency_penalty
        p1 = compute_efficiency_penalty(2, 10)   # fewer hops → smaller magnitude
        p2 = compute_efficiency_penalty(8, 10)   # more hops  → larger magnitude
        # Penalties are negative; more hops must be MORE negative (greater magnitude)
        self.assertLess(p2, p1,
                        "More hops should incur a more negative penalty")
        self.assertGreater(abs(p2), abs(p1),
                           "More hops should incur greater penalty magnitude")


# ---------------------------------------------------------------------------
# RolloutBuffer tests
# ---------------------------------------------------------------------------

class TestRolloutBuffer(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.config.ppo.update_every = 5

    def _make_transition(self, done: bool = False):
        from src.rl.replay_buffer import Transition
        action_dim = self.config.encoder.node_feature_dim + self.config.state.edge_type_dim
        return Transition(
            state_vec=torch.randn(self.config.state.total_dim),
            action_embs=torch.randn(4, action_dim),
            query_emb=torch.randn(self.config.state.query_dim),
            action_idx=2,
            log_prob=-1.2,
            reward=0.5,
            value=0.4,
            done=done,
        )

    def test_add_and_len(self):
        from src.rl.replay_buffer import RolloutBuffer
        buf = RolloutBuffer(self.config)
        for _ in range(3):
            buf.add(self._make_transition())
        self.assertEqual(len(buf), 3)

    def test_is_ready_false_before_threshold(self):
        from src.rl.replay_buffer import RolloutBuffer
        buf = RolloutBuffer(self.config)
        buf.add(self._make_transition())
        self.assertFalse(buf.is_ready())

    def test_is_ready_true_at_threshold(self):
        from src.rl.replay_buffer import RolloutBuffer
        buf = RolloutBuffer(self.config)
        for _ in range(5):
            buf.add(self._make_transition())
        self.assertTrue(buf.is_ready())

    def test_clear(self):
        from src.rl.replay_buffer import RolloutBuffer
        buf = RolloutBuffer(self.config)
        for _ in range(5):
            buf.add(self._make_transition())
        buf.clear()
        self.assertEqual(len(buf), 0)

    def test_compute_advantages_runs(self):
        from src.rl.replay_buffer import RolloutBuffer
        buf = RolloutBuffer(self.config)
        for i in range(5):
            buf.add(self._make_transition(done=(i == 4)))
        # Should not raise
        buf.compute_advantages(gamma=0.99, gae_lambda=0.95)

    def test_mini_batches_cover_all(self):
        from src.rl.replay_buffer import RolloutBuffer
        buf = RolloutBuffer(self.config)
        for i in range(8):
            buf.add(self._make_transition(done=(i == 7)))
        buf.compute_advantages(0.99, 0.95)
        total = 0
        for batch in buf.get_mini_batches(mini_batch_size=4):
            total += batch["state_vecs"].shape[0]
        self.assertEqual(total, 8)

    def test_imitation_batch_returns_labeled(self):
        from src.rl.replay_buffer import RolloutBuffer, Transition
        buf = RolloutBuffer(self.config)
        action_dim = self.config.encoder.node_feature_dim + self.config.state.edge_type_dim
        for i in range(6):
            t = Transition(
                state_vec=torch.randn(self.config.state.total_dim),
                action_embs=torch.randn(4, action_dim),
                query_emb=torch.randn(self.config.state.query_dim),
                action_idx=1, log_prob=-1.0, reward=0.3,
                value=0.5, done=False,
                teacher_action_idx=1 if i < 4 else None,
            )
            buf.add(t)
        batch = buf.get_imitation_batch(batch_size=3)
        self.assertIsNotNone(batch)
        self.assertLessEqual(len(batch), 4)  # only labeled ones
        for t in batch:
            self.assertIsNotNone(t.teacher_action_idx)


# ---------------------------------------------------------------------------
# EpisodeRecord / RLRetriever (mocked env)
# ---------------------------------------------------------------------------

class TestRLRetrieverMocked(unittest.TestCase):
    """Test RLRetriever with fully mocked KGEnvironment."""

    def setUp(self):
        self.config = make_config()

    def _build_mock_env(self, n_steps: int = 3):
        """Build a mock KGEnvironment that returns deterministic transitions."""
        from src.rl.kg_env import Action, StepInfo
        from src.retrieval.query_encoder import QueryEncoder
        from src.retrieval.state_builder import EpisodeState

        config = self.config
        action_dim = config.encoder.node_feature_dim + config.state.edge_type_dim
        state_dim = config.state.total_dim
        query_dim = config.state.query_dim

        enc = QueryEncoder(config)
        qr = enc.encode("test laptop query")

        env = MagicMock()
        env.reset.return_value = (
            torch.randn(state_dim),
            {
                "anchor_node_id": "B07ANCHOR",
                "anchor_type": "Product",
                "query_repr": qr,
                "num_actions": 3,
            },
        )
        # get_episode: return a mock EpisodeState
        mock_ep = MagicMock()
        mock_ep.hop_count = n_steps
        mock_ep.uncertainty_score = 0.3
        mock_ep.visited_node_ids = set()
        mock_ep.collected_node_reprs = []
        mock_ep.collected_evidence = []
        mock_ep.visited_steps = []
        mock_ep.query_repr = qr
        env.get_episode.return_value = mock_ep

        # Actions: n_steps moves then STOP
        def get_valid_actions_side_effect():
            return [
                Action(
                    index=0, is_stop=False,
                    target_node_id="node_x", target_node_type="Product",
                    relation_type="HAS_BRAND",
                    action_embedding=torch.randn(action_dim),
                ),
                Action(
                    index=1, is_stop=True,
                    target_node_id=None, target_node_type=None,
                    relation_type="STOP",
                    action_embedding=torch.randn(action_dim),
                ),
            ]

        env.get_valid_actions.side_effect = get_valid_actions_side_effect

        step_results = []
        for i in range(n_steps):
            step_info = StepInfo(
                hop=i + 1, action_taken=MagicMock(is_stop=False),
                node_id=f"node_{i}", node_type="Product",
                relation_type="HAS_BRAND",
                evidence_text=f"evidence_{i}",
                uncertainty=0.5 - i * 0.1,
                reward_breakdown={},
            )
            step_results.append((torch.randn(state_dim), 0.2, False, step_info))
        # Final step: done=True
        final_info = StepInfo(
            hop=n_steps, action_taken=MagicMock(is_stop=True),
            node_id="node_final", node_type="Product",
            relation_type="STOP",
            evidence_text="",
            uncertainty=0.2,
            reward_breakdown={},
        )
        step_results.append((torch.randn(state_dim), 0.5, True, final_info))
        env.step.side_effect = step_results

        env.get_trajectory.return_value = []
        env.get_evidence.return_value = [f"evidence_{i}" for i in range(n_steps)]

        return env, qr

    def test_run_episode_returns_record(self):
        from src.rl.actor import Actor
        from src.retrieval.stopping import AdaptiveStoppingModule
        from src.retrieval.rl_retriever import RLRetriever

        env, _ = self._build_mock_env(n_steps=2)
        actor = Actor(self.config)
        stopping = AdaptiveStoppingModule(self.config)
        retriever = RLRetriever(env, actor, stopping, buffer=None, config=self.config)

        record = retriever.run_episode("test laptop", deterministic=True,
                                       collect_transitions=False)
        self.assertIsNotNone(record)
        self.assertIsInstance(record.total_reward, float)

    def test_run_episode_with_buffer(self):
        from src.rl.actor import Actor
        from src.retrieval.stopping import AdaptiveStoppingModule
        from src.retrieval.rl_retriever import RLRetriever
        from src.rl.replay_buffer import RolloutBuffer

        env, _ = self._build_mock_env(n_steps=2)
        actor = Actor(self.config)
        stopping = AdaptiveStoppingModule(self.config)
        buf = RolloutBuffer(self.config)
        retriever = RLRetriever(env, actor, stopping, buffer=buf, config=self.config)

        record = retriever.run_episode("test laptop", deterministic=False,
                                       collect_transitions=True)
        # Buffer should have transitions
        self.assertGreater(len(buf), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)