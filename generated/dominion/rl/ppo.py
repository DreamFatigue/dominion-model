"""PPO clipped-surrogate update against a RolloutBuffer.

Loss is computed via DominionPolicy.evaluate_actions, which already groups
a mixed-kind minibatch by info["kind"] internally and dispatches to the
right distribution (categorical+STOP for action/buy/pile, independent
Bernoulli for cards) -- ppo.py itself stays kind-agnostic, it just supplies
a minibatch and reads back per-sample log_prob/entropy/value.
"""

import numpy as np
import torch
import torch.nn as nn


class PolicyLossError(RuntimeError):
    """Raised instead of silently continuing on a NaN/Inf loss -- a
    corrupted checkpoint from a bad update is far more expensive to notice
    later than failing loudly right here (see the plan's verification step 3)."""


def ppo_update(policy, optimizer, buffer, clip_eps=0.2, epochs=4, minibatch_size=256,
               vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    hand, piles, player_state = buffer.stacked_obs_tensors()
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
    advantages = torch.tensor(buffer.advantages, dtype=torch.float32)
    returns = torch.tensor(buffer.returns, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(buffer)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]
            mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long)

            mb_kinds = [buffer.kinds[i] for i in mb_idx]
            mb_masks = [buffer.masks[i] for i in mb_idx]
            mb_actions = [buffer.actions[i] for i in mb_idx]

            new_log_prob, entropy, value = policy.evaluate_actions(
                hand[mb_idx_t], piles[mb_idx_t], player_state[mb_idx_t], mb_kinds, mb_masks, mb_actions)

            mb_old_log_prob = old_log_probs[mb_idx_t]
            mb_advantages = advantages[mb_idx_t]
            mb_returns = returns[mb_idx_t]

            ratio = torch.exp(new_log_prob - mb_old_log_prob)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(value, mb_returns)
            entropy_mean = entropy.mean()

            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise PolicyLossError(
                    f"total_loss is {total_loss.item()} (policy_loss={policy_loss.item()}, "
                    f"value_loss={value_loss.item()}, entropy={entropy_mean.item()}) -- aborting "
                    "before this corrupts the policy weights."
                )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_log_prob - new_log_prob).mean().item()
                clip_fraction = (torch.abs(ratio - 1.0) > clip_eps).float().mean().item()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy_mean.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_fraction"].append(clip_fraction)

    return {k: float(np.mean(v)) for k, v in stats.items()}
