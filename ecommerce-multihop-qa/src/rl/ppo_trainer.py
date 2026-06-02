"""
src/rl/ppo_trainer.py

PPO Trainer — Stage 2 of the two-stage training pipeline.

Implements Proximal Policy Optimisation (PPO-clip) for the adaptive
retrieval policy.  Designed specifically for the sequential KG traversal
setting where:
  - Action spaces vary in size across steps (padded in RolloutBuffer)
  - Episodes have variable length (adaptive stopping)
  - The policy is query-aware (query embedding gates attention)

Training loop
-------------
1. Collect N_rollout steps across episodes → RolloutBuffer
2. Compute GAE advantages
3. For ppo_epochs: iterate over mini-batches
     a. Re-evaluate (state, action) under current policy
     b. Compute PPO clip loss + value loss + entropy bonus
     c. Gradient step
4. Log metrics, save checkpoint if improved
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.config.settings import Config, DEFAULT_CONFIG
from src.rl.actor import Actor
from src.rl.critic import Critic
from src.rl.replay_buffer import RolloutBuffer, Transition
from src.rl.kg_env import KGEnvironment
from src.rl.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO trainer for the adaptive retrieval policy.

    Parameters
    ----------
    actor   : Actor
    critic  : Critic  (utility class; value head is inside actor.policy)
    env     : KGEnvironment
    buffer  : RolloutBuffer
    config  : Config
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        env: KGEnvironment,
        buffer: RolloutBuffer,
        config: Config = DEFAULT_CONFIG,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.env = env
        self.buffer = buffer
        self.config = config
        self.device = torch.device(config.encoder.device)
        self.ckpt = checkpoint_manager

        ppo = config.ppo
        # Single optimiser for actor (which contains both actor + critic heads)
        self.optimiser = optim.Adam(
            actor.get_parameters(),
            lr=ppo.lr_actor,
            eps=1e-5,
        )
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimiser,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=1000,
        )

        self._global_step: int = 0
        self._best_reward: float = float("-inf")
        logger.info("PPOTrainer initialised.")

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        queries: List[str],
        n_iterations: int = 200,
        anchor_ids: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, List[float]]:
        """
        Full PPO training loop.

        Parameters
        ----------
        queries      : list of training query strings
        n_iterations : number of collect+update cycles
        anchor_ids   : optional per-query anchor node IDs

        Returns
        -------
        metrics : dict of logged metric lists
        """
        metrics: Dict[str, List[float]] = {
            "iteration_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }

        for iteration in range(n_iterations):
            t0 = time.time()

            # === Phase 1: Collect rollouts ===
            collect_stats = self._collect_rollouts(
                queries=queries,
                anchor_ids=anchor_ids,
            )

            # === Phase 2: Compute advantages ===
            self.buffer.compute_advantages(
                gamma=self.config.ppo.gamma,
                gae_lambda=self.config.ppo.gae_lambda,
            )

            # === Phase 3: PPO update ===
            update_stats = self._ppo_update()

            # === Logging ===
            elapsed = time.time() - t0
            mean_reward = collect_stats.get("mean_episode_reward", 0.0)
            metrics["iteration_reward"].append(mean_reward)
            metrics["policy_loss"].append(update_stats.get("policy_loss", 0.0))
            metrics["value_loss"].append(update_stats.get("value_loss", 0.0))
            metrics["entropy"].append(update_stats.get("entropy", 0.0))
            metrics["approx_kl"].append(update_stats.get("approx_kl", 0.0))

            logger.info(
                "Iter %3d/%d | reward=%.4f | ploss=%.4f | vloss=%.4f | "
                "entropy=%.4f | kl=%.4f | t=%.1fs",
                iteration + 1, n_iterations,
                mean_reward,
                update_stats.get("policy_loss", 0.0),
                update_stats.get("value_loss", 0.0),
                update_stats.get("entropy", 0.0),
                update_stats.get("approx_kl", 0.0),
                elapsed,
            )

            # Checkpoint
            if mean_reward > self._best_reward:
                self._best_reward = mean_reward
                if self.ckpt is not None:
                    self.ckpt.save(
                        model=self.actor.policy,
                        optimiser=self.optimiser,
                        step=self._global_step,
                        metrics={"best_reward": mean_reward},
                        tag="best",
                    )

            # Clear buffer for next iteration
            self.buffer.clear()
            self.scheduler.step()

        return metrics

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(
        self,
        queries: List[str],
        anchor_ids: Optional[List[Optional[str]]],
    ) -> Dict[str, float]:
        """
        Run the current policy in the environment to fill the buffer.
        Continues until buffer.is_ready() returns True.
        """
        self.actor.policy.eval()
        total_reward = 0.0
        n_episodes = 0
        steps_collected = 0

        query_pool = list(queries)
        import random
        random.shuffle(query_pool)
        q_cycle = iter(query_pool * 100)   # enough repetitions

        while not self.buffer.is_ready():
            query = next(q_cycle)
            anchor = None
            if anchor_ids:
                idx = queries.index(query) if query in queries else 0
                anchor = anchor_ids[idx] if idx < len(anchor_ids) else None

            episode_reward = self._run_episode(query, anchor)
            total_reward += episode_reward
            n_episodes += 1
            steps_collected += 1

        return {
            "mean_episode_reward": total_reward / max(n_episodes, 1),
            "n_episodes": n_episodes,
        }

    def _run_episode(
        self, query: str, anchor_id: Optional[str] = None
    ) -> float:
        """Run one complete episode and store transitions in the buffer."""
        obs, info = self.env.reset(query, anchor_id)
        query_emb = info["query_repr"].embedding

        self.buffer.start_episode()
        episode_reward = 0.0
        done = False

        while not done:
            actions = self.env.get_valid_actions()
            action_idx, log_prob, value, _ = self.actor.act(
                state_vec=obs,
                actions=actions,
                query_emb=query_emb,
                deterministic=False,
            )
            obs_next, reward, done, step_info = self.env.step(action_idx)

            # Build action_embs tensor for this step
            from src.rl.actor import Actor
            action_embs, _ = self.actor._stack_action_embeddings(actions)

            transition = Transition(
                state_vec=obs.detach().cpu(),
                action_embs=action_embs.detach().cpu(),
                query_emb=query_emb.detach().cpu(),
                action_idx=action_idx,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
                hop=step_info.hop,
                node_id=step_info.node_id,
                node_type=step_info.node_type,
                relation=step_info.relation_type,
                uncertainty=step_info.uncertainty,
                reward_breakdown=step_info.reward_breakdown,
            )
            self.buffer.add(transition)
            episode_reward += reward
            obs = obs_next
            self._global_step += 1

        self.buffer.end_episode()
        return episode_reward

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> Dict[str, float]:
        """
        Perform ppo_epochs of mini-batch gradient updates.

        Returns aggregated loss statistics.
        """
        self.actor.policy.train()
        ppo = self.config.ppo
        stats: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [],
            "entropy": [], "approx_kl": [],
        }

        for epoch in range(ppo.ppo_epochs):
            for batch in self.buffer.get_mini_batches(ppo.mini_batch_size):

                # Re-evaluate actions under current policy
                new_log_probs, entropy, new_values = self.actor.evaluate_actions(
                    state_vecs=batch["state_vecs"],
                    action_embs_batch=batch["action_embs"],
                    query_embs=batch["query_embs"],
                    action_indices=batch["action_indices"],
                    action_masks=batch["action_masks"],
                )

                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                # --- Policy (actor) loss ---
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = ratio.clamp(
                    1.0 - ppo.clip_epsilon, 1.0 + ppo.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value (critic) loss ---
                value_loss = self.critic.value_loss(
                    predicted_values=new_values,
                    returns=returns,
                    old_values=old_values,
                    clip_range=ppo.clip_epsilon,
                )

                # --- Entropy bonus ---
                entropy_loss = -entropy.mean()

                # --- Total loss ---
                loss = (
                    policy_loss
                    + ppo.value_loss_coef * value_loss
                    + ppo.entropy_coef * entropy_loss
                )

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.get_parameters(), ppo.max_grad_norm
                )
                self.optimiser.step()

                # Approximate KL divergence (for early stopping)
                with torch.no_grad():
                    approx_kl = ((old_log_probs - new_log_probs) ** 2).mean().item() * 0.5

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(-entropy_loss.item())
                stats["approx_kl"].append(approx_kl)

                # Early exit if KL too large
                if approx_kl > 0.02:
                    logger.debug("Early KL stop at epoch %d", epoch)
                    break

        return {k: sum(v) / max(len(v), 1) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Imitation Learning Trainer (Stage 1)
# ---------------------------------------------------------------------------

class ImitationTrainer:
    """
    Stage 1: warm-up the policy via behavioural cloning on high-quality trajectories.

    Parameters
    ----------
    actor  : Actor
    buffer : RolloutBuffer  (pre-filled with labelled trajectories)
    config : Config
    """

    def __init__(
        self,
        actor: Actor,
        buffer: RolloutBuffer,
        config: Config = DEFAULT_CONFIG,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.actor = actor
        self.buffer = buffer
        self.config = config
        self.ckpt = checkpoint_manager
        self.device = torch.device(config.encoder.device)

        ic = config.imitation
        self.optimiser = optim.Adam(actor.get_parameters(), lr=ic.lr)
        logger.info("ImitationTrainer initialised.")

    def train(self, n_epochs: int = None) -> List[float]:
        """
        Train for n_epochs over the labelled buffer.

        Returns
        -------
        epoch_losses : list of per-epoch mean loss
        """
        ic = self.config.imitation
        n_epochs = n_epochs or ic.epochs
        epoch_losses: List[float] = []

        for epoch in range(n_epochs):
            batch = self.buffer.get_imitation_batch(ic.batch_size)
            if batch is None:
                logger.warning("No labelled transitions in buffer. Stopping imitation.")
                break

            self.actor.policy.train()
            total_loss = 0.0
            for trans in batch:
                from src.rl.actor import Actor  # avoid circular at top-level
                # Re-build action list context is not available; use stored embs
                # Compute cross-entropy directly on stored embeddings
                logits, probs, _, _ = self.actor.policy(
                    state_vec=trans.state_vec.to(self.device),
                    action_embs=trans.action_embs.to(self.device),
                    query_emb=trans.query_emb.to(self.device),
                )
                target = torch.tensor(
                    trans.teacher_action_idx, dtype=torch.long, device=self.device
                )
                loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), target.unsqueeze(0)
                )
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.get_parameters(),
                    self.config.ppo.max_grad_norm,
                )
                self.optimiser.step()
                total_loss += loss.item()

            mean_loss = total_loss / max(len(batch), 1)
            epoch_losses.append(mean_loss)
            logger.info("Imitation epoch %d/%d — loss=%.4f", epoch + 1, n_epochs, mean_loss)

            if self.ckpt and epoch % 5 == 0:
                self.ckpt.save(
                    model=self.actor.policy,
                    optimiser=self.optimiser,
                    step=epoch,
                    metrics={"imitation_loss": mean_loss},
                    tag="imitation",
                )

        return epoch_losses