"""Policy network for PPO training against DominionEnv.

Token-based architecture, weight-shared across slot position -- never
slot-index-specific -- because action_space.assign_pile_slots deliberately
randomizes pile->slot mapping every episode specifically to stop the policy
overfitting to "slot 3 is always Village" (see action_space.py); any
per-slot-position weight would silently reintroduce exactly that.

The 4 decision kinds (info["kind"] in env.py) share their underlying
per-slot scoring functions rather than getting 4 separate heads:
- "action" (categorical+STOP over hand slots) and "cards" (independent
  per-slot Bernoulli, no STOP) both ask "how good is playing/selecting the
  card in slot i" and never co-occur at the same step, so they share
  hand_scorer.
- "buy" and "pile" (both categorical+STOP over pile slots) likewise share
  pile_scorer.
Splitting them into separate heads would just halve each head's training
signal for no benefit.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Independent

from .features import FEATURE_DIM

PILE_FEATURE_DIM = FEATURE_DIM + 1  # card features + trailing pile count
MASK_NEG = -1e9

CATEGORICAL_KINDS = ("action", "buy", "pile")


def _mlp(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim), nn.ReLU(),
    )


def _presence_mask(x):
    """A real card's feature vector is never all-zero (every card has at
    least one CardTypeKind flag set to 1.0); padding rows are exactly zero.
    So a nonzero check is a reliable, env-independent way to tell real slots
    from padding for pooling, without needing the decision-specific legality
    mask (which only covers *choosable* slots, not the whole hand/supply)."""
    return x.abs().sum(dim=-1) > 1e-8


def _masked_mean(tokens, presence):
    presence_f = presence.float().unsqueeze(-1)
    summed = (tokens * presence_f).sum(dim=1)
    counts = presence_f.sum(dim=1).clamp(min=1.0)
    return summed / counts


class _SlotScorer(nn.Module):
    """Per-slot logit, context-aware: concatenates the global context onto
    every token before scoring, so e.g. "is this card worth playing" can
    depend on current_priority/deck_strength, not just the card in isolation."""

    def __init__(self, token_dim, context_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim + context_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens, context):
        context_expanded = context.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        x = torch.cat([tokens, context_expanded], dim=-1)
        return self.net(x).squeeze(-1)


class DominionPolicy(nn.Module):
    def __init__(self, token_dim=64, context_dim=128, hidden=64):
        super().__init__()
        self.hand_encoder = _mlp(FEATURE_DIM, token_dim, hidden)
        self.pile_encoder = _mlp(PILE_FEATURE_DIM, token_dim, hidden)
        self.player_state_encoder = _mlp(4, token_dim, hidden)
        self.context_mlp = _mlp(token_dim * 3, context_dim, hidden)

        self.hand_scorer = _SlotScorer(token_dim, context_dim, hidden)
        self.pile_scorer = _SlotScorer(token_dim, context_dim, hidden)
        self.hand_stop_head = nn.Linear(context_dim, 1)
        self.pile_stop_head = nn.Linear(context_dim, 1)
        self.value_head = nn.Linear(context_dim, 1)

    def forward(self, hand, piles, player_state):
        """hand: (B, HAND_SLOTS_MAX, FEATURE_DIM), piles: (B, PILE_SLOTS,
        PILE_FEATURE_DIM), player_state: (B, 4). Returns (hand_logits (B,
        HAND_SLOTS_MAX), pile_logits (B, PILE_SLOTS), hand_stop (B,1),
        pile_stop (B,1), value (B,))."""
        hand_tokens = self.hand_encoder(hand)
        pile_tokens = self.pile_encoder(piles)

        hand_pool = _masked_mean(hand_tokens, _presence_mask(hand))
        pile_pool = _masked_mean(pile_tokens, _presence_mask(piles))
        ps_emb = self.player_state_encoder(player_state)
        context = self.context_mlp(torch.cat([hand_pool, pile_pool, ps_emb], dim=-1))

        hand_logits = self.hand_scorer(hand_tokens, context)
        pile_logits = self.pile_scorer(pile_tokens, context)
        hand_stop = self.hand_stop_head(context)
        pile_stop = self.pile_stop_head(context)
        value = self.value_head(context).squeeze(-1)
        return hand_logits, pile_logits, hand_stop, pile_stop, value

    def _kind_logits(self, kind, hand_logits, pile_logits, hand_stop, pile_stop):
        if kind == "action":
            return torch.cat([hand_logits, hand_stop], dim=-1)
        if kind in ("buy", "pile"):
            return torch.cat([pile_logits, pile_stop], dim=-1)
        if kind == "cards":
            return hand_logits
        raise ValueError(kind)

    def _make_dist(self, kind, logits, mask):
        masked_logits = logits.masked_fill(~mask, MASK_NEG)
        if kind == "cards":
            return Independent(Bernoulli(logits=masked_logits), 1)
        return Categorical(logits=masked_logits)

    @torch.no_grad()
    def act(self, obs, mask, kind, deterministic=False):
        """Single-observation inference (batch size 1). Returns (action,
        log_prob: float, value: float) where `action` is already in the
        exact shape env.py's step() expects for this `kind` (int for
        action/buy/pile, a length-HAND_SLOTS_MAX list of 0/1 for cards)."""
        hand = torch.as_tensor(np.asarray(obs["hand"]), dtype=torch.float32).unsqueeze(0)
        piles = torch.as_tensor(np.asarray(obs["piles"]), dtype=torch.float32).unsqueeze(0)
        player_state = torch.as_tensor(np.asarray(obs["player_state"]), dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool).unsqueeze(0)

        hand_logits, pile_logits, hand_stop, pile_stop, value = self.forward(hand, piles, player_state)
        logits = self._kind_logits(kind, hand_logits, pile_logits, hand_stop, pile_stop)
        dist = self._make_dist(kind, logits, mask_t)

        if deterministic:
            masked_logits = logits.masked_fill(~mask_t, MASK_NEG)
            action_t = (masked_logits > 0).float() if kind == "cards" else torch.argmax(masked_logits, dim=-1)
        else:
            action_t = dist.sample()
        log_prob = dist.log_prob(action_t)

        if kind == "cards":
            action = [int(b) for b in action_t[0].tolist()]
        else:
            action = int(action_t[0].item())
        return action, float(log_prob.item()), float(value.item())

    def evaluate_actions(self, hand, piles, player_state, kinds, masks, actions):
        """Batched, for PPO updates. hand/piles/player_state: uniform-shape
        tensors, (B, ...) each. kinds: list[str] length B. masks: list of
        1-D bool arrays length B (variable width per-sample depending on
        kind -- HAND_SLOTS_MAX+1 for action, PILE_SLOTS+1 for buy/pile,
        HAND_SLOTS_MAX for cards -- so grouped by kind before stacking).
        actions: list length B, matching each sample's `act()` output shape.
        Returns (log_prob (B,), entropy (B,), value (B,))."""
        hand_logits, pile_logits, hand_stop, pile_stop, value = self.forward(hand, piles, player_state)
        B = hand.shape[0]
        device = hand.device
        log_prob = torch.zeros(B, device=device)
        entropy = torch.zeros(B, device=device)
        kinds_arr = np.array(kinds)

        for kind in CATEGORICAL_KINDS + ("cards",):
            idx = np.nonzero(kinds_arr == kind)[0]
            if len(idx) == 0:
                continue
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
            sub_mask = torch.as_tensor(np.stack([masks[i] for i in idx]), dtype=torch.bool, device=device)
            logits = self._kind_logits(
                kind, hand_logits[idx_t], pile_logits[idx_t], hand_stop[idx_t], pile_stop[idx_t])
            dist = self._make_dist(kind, logits, sub_mask)
            if kind == "cards":
                act_t = torch.as_tensor(np.stack([actions[i] for i in idx]), dtype=torch.float32, device=device)
            else:
                act_t = torch.as_tensor([actions[i] for i in idx], dtype=torch.long, device=device)
            log_prob[idx_t] = dist.log_prob(act_t)
            entropy[idx_t] = dist.entropy()

        return log_prob, entropy, value
