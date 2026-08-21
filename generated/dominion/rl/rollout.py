"""Rollout collection + GAE(lambda) for PPO training against DominionEnv."""

import numpy as np
import torch

GAMMA_DEFAULT = 0.99
LAMBDA_DEFAULT = 0.95


class RolloutBuffer:
    """List-based accumulation -- episode length isn't known ahead of time,
    so this doesn't preallocate. Converted to stacked tensors only when
    actually needed (obs tensors in ppo.py's minibatching); every obs/mask
    already has a uniform per-kind shape thanks to env.py's fixed-shape
    padding, so stacking is a plain np.stack, no ragged handling."""

    def __init__(self):
        self.obs = []
        self.kinds = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def add(self, obs, kind, mask, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.kinds.append(kind)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)

    def stacked_obs_tensors(self):
        hand = torch.as_tensor(np.stack([o["hand"] for o in self.obs]), dtype=torch.float32)
        piles = torch.as_tensor(np.stack([o["piles"] for o in self.obs]), dtype=torch.float32)
        player_state = torch.as_tensor(np.stack([o["player_state"] for o in self.obs]), dtype=torch.float32)
        return hand, piles, player_state


def compute_gae(rewards, values, dones, bootstrap_value, gamma=GAMMA_DEFAULT, lam=LAMBDA_DEFAULT):
    """dones[t] == True means step t was the last step of an episode (the
    *next* stored transition, if any, starts a fresh episode) -- the running
    advantage accumulator is implicitly reset at every such boundary, since
    next_non_terminal zeroes out continuity across it."""
    T = len(rewards)
    advantages = [0.0] * T
    last_gae_lam = 0.0
    for t in reversed(range(T)):
        next_value = bootstrap_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns


def collect_rollout(env, policy, obs, info, n_steps, gamma=GAMMA_DEFAULT, lam=LAMBDA_DEFAULT):
    """Steps the *live* env for a fixed step budget, resuming from whatever
    (obs, info) the caller passes in rather than forcing a reset -- so
    consecutive calls across training iterations keep stepping the same
    live game/background thread, only resetting when a transition actually
    terminates. Returns (buffer, obs, info) so the caller can feed the pair
    straight into the next collect_rollout call."""
    buffer = RolloutBuffer()
    for _ in range(n_steps):
        kind, mask = info["kind"], info["mask"]
        action, log_prob, value = policy.act(obs, mask, kind, deterministic=False)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        buffer.add(obs, kind, mask, action, log_prob, value, reward, done)
        obs, info = next_obs, next_info
        if done:
            obs, info = env.reset()

    if buffer.dones[-1]:
        bootstrap_value = 0.0
    else:
        with torch.no_grad():
            hand = torch.as_tensor(np.asarray(obs["hand"]), dtype=torch.float32).unsqueeze(0)
            piles = torch.as_tensor(np.asarray(obs["piles"]), dtype=torch.float32).unsqueeze(0)
            player_state = torch.as_tensor(np.asarray(obs["player_state"]), dtype=torch.float32).unsqueeze(0)
            _, _, _, _, value_t = policy.forward(hand, piles, player_state)
        bootstrap_value = float(value_t.item())

    buffer.advantages, buffer.returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones, bootstrap_value, gamma=gamma, lam=lam)
    return buffer, obs, info
