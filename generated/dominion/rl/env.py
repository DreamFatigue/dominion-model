"""Section 6: Gym-Style Env Wrapper.

The real game engine (player.py) calls choose_*_fn synchronously, including
deeply nested calls (Mine's trash-then-gain, Militia landing on whichever
opponent it's attacking). To turn that into a step()-able env without
rewriting the engine, one seat ("hero", always game.players[0]) gets hooks
that block on a queue handoff instead of returning immediately; the actual
game runs to completion on a background thread. Opponents keep their normal
synchronous baseline hooks and never block anything.

Because hero's hooks are wired at the Player level (not scoped to "is it
hero's turn"), an opponent's Militia landing on hero mid-opponent-turn hits
the same blocking hook and correctly yields control to the RL caller with
hero's own PlayerState/mask -- off-turn routing (Section 6) falls out of
this design for free, without any separate "whose turn is it" bookkeeping.

Simplification vs. the plan's literal description: choose_cards_fn's
"toggle-then-commit sequence" (Section 3) is collapsed into a single
MultiBinary(HAND_SLOTS) action per call, decoded to the full chosen subset
at once, rather than multiple discrete toggle/STOP steps. Same legal
selections are reachable; it's a one-shot multi-select instead of a
multi-step one. Kept this way for implementation tractability -- flagged
here rather than silently deviating from Section 3.
"""

import queue
import random
import threading

import gymnasium as gym
import numpy as np

from ..structure.game import Game
from ..cards.cardtypekind import CardTypeKind
from .state import compute_player_state, capture_initial_pile_counts, Priority
from .action_space import (
    compute_hand_slots, assign_pile_slots, pile_slot_mask, decode_pile_choice,
    hand_slot_mask, decode_hand_choice, multi_select_mask, decode_multi_select,
    PILE_SLOTS,
)
from .features import card_feature_vector, pile_feature_vector, FEATURE_DIM
from .reward import compute_reward
from .agents import set_heuristic_baseline
from .play_style import PlayStyleLog

MAX_ROUNDS = 300  # matches demo.py's spirit of a hard cap against runaway games


class _DecisionRequest:
    __slots__ = ("kind", "candidates", "prompt", "context", "count")

    def __init__(self, kind, candidates, prompt, context, count=None):
        self.kind = kind
        self.candidates = candidates
        self.prompt = prompt
        self.context = context
        self.count = count


class DominionEnv(gym.Env):
    """One env-controlled seat (hero, seat 0) vs. baseline-controlled
    opponents. 2 players by default, matching Section 5's starting point."""

    metadata = {"render_modes": []}

    def __init__(self, num_players=2, opponent_setup=set_heuristic_baseline, seed=None):
        super().__init__()
        self.num_players = num_players
        self.opponent_setup = opponent_setup
        self._py_rng = random.Random(seed)

        self.observation_space = gym.spaces.Dict({
            "hand": gym.spaces.Box(low=-1, high=1, shape=(1, FEATURE_DIM), dtype=np.float32),
            "piles": gym.spaces.Box(low=-1, high=1, shape=(PILE_SLOTS, FEATURE_DIM + 1), dtype=np.float32),
            "player_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        })
        # Actual per-step action space depends on which of the 4 decision
        # kinds is pending (see info["kind"]/info["mask"]); Discrete(1) here
        # is just a placeholder satisfying gym.Env's interface.
        self.action_space = gym.spaces.Discrete(1)

        self.game = None
        self.hero = None
        self._thread = None
        self._request_q = None
        self._reply_q = None
        self._pending_request = None
        self._prev_player_state = None
        self._done = False
        self.play_style_log = None

    # -- gym.Env interface -----------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._py_rng = random.Random(seed)

        self.game = Game()
        self.game.setup(self.num_players)
        self.hero = self.game.players[0]
        for opp in self.game.players[1:]:
            self.opponent_setup(opp)

        self.hero.choose_action_fn = self._make_hook("action")
        self.hero.choose_buy_fn = self._make_hook("buy")
        self.hero.choose_cards_fn = self._make_hook("cards")
        self.hero.choose_pile_fn = self._make_hook("pile")

        self.hand_slots = compute_hand_slots(self.game.supply)
        self.slot_assignment = assign_pile_slots(self.game.supply, rng=self._py_rng)
        self.initial_pile_counts = capture_initial_pile_counts(self.game.supply)
        self.round_num = 0
        self.play_style_log = PlayStyleLog()
        self._done = False

        self._request_q = queue.Queue(maxsize=1)
        self._reply_q = queue.Queue(maxsize=1)

        self._prev_player_state = compute_player_state(
            self.hero, self.game.players, self.game.supply, self.initial_pile_counts, self.round_num)

        self._thread = threading.Thread(target=self._run_game, daemon=True)
        self._thread.start()

        request = self._request_q.get()
        if request is None:
            # Degenerate case: game ended before hero ever got a decision.
            return self._terminal_obs(), {"mask": np.zeros(1, dtype=bool), "kind": None, "done": True}
        self._pending_request = request
        obs, mask = self._build_obs(request)
        return obs, {"mask": mask, "kind": request.kind}

    def step(self, action):
        request = self._pending_request
        decoded = self._decode_action(request, action)
        self._reply_q.put(decoded)

        next_request = self._request_q.get()

        next_state = compute_player_state(
            self.hero, self.game.players, self.game.supply, self.initial_pile_counts, self.round_num)
        reward = compute_reward(self._prev_player_state, next_state, done=self._done and next_request is None)
        self._prev_player_state = next_state

        terminated = next_request is None
        info = {"score_gap": next_state.score_gap, "deck_strength": next_state.deck_strength}
        if terminated:
            self._thread.join(timeout=5)
            return self._terminal_obs(), reward, True, False, info

        self._pending_request = next_request
        obs, mask = self._build_obs(next_request)
        info["mask"] = mask
        info["kind"] = next_request.kind
        return obs, reward, False, False, info

    # -- background game thread -------------------------------------------

    def _make_hook(self, kind):
        def hook(candidates, prompt, context, count=None):
            self._request_q.put(_DecisionRequest(kind, candidates, prompt, context, count))
            return self._reply_q.get()
        return hook

    def _run_game(self):
        try:
            while not self.game.is_over() and self.round_num < MAX_ROUNDS:
                self.round_num += 1
                for p in self.game.players:
                    self._play_turn(p)
                    if self.game.is_over():
                        break
        finally:
            self._done = True
            self._request_q.put(None)

    def _play_turn(self, p):
        # Computed once, up front: seeds p.current_priority (which the new
        # ai_choose_cards heuristics read) and, for hero, is the exact value
        # logged into play_style_log -- one computation drives both, so they
        # can't drift apart.
        ps = compute_player_state(
            p, self.game.players, self.game.supply, self.initial_pile_counts, self.round_num)
        p.current_priority = ps.current_priority
        if p is self.hero:
            self.play_style_log.record(ps.current_priority)

        while p.actions > 0:
            action_cards = [c for c in p.hand.cards if CardTypeKind.Action in c.types]
            if not action_cards:
                break
            card = p.choose_action_fn(action_cards, "Choose an action card to play", "action_phase")
            if card is None:
                break
            p.play_action_card(card, self.game)
        p.play_treasures()
        while p.buys > 0:
            affordable = [pl for pl in self.game.supply.piles if pl.count > 0 and pl.cardType.cost <= p.coins]
            if not affordable:
                break
            best = p.choose_buy_fn(affordable, "Choose a pile to buy", "buy_phase")
            if best is None:
                break
            p.buy_card(best)
        p.end_turn()

    # -- obs / action (de)coding -------------------------------------------

    def _build_obs(self, request):
        hand = self.hero.hand.cards
        owned = len(hand) + len(self.hero.deck.cards) + len(self.hero.discardPile.cards) + len(self.hero.playArea.cards)

        hand_feats = np.zeros((max(1, min(len(hand), self.hand_slots)), FEATURE_DIM), dtype=np.float32)
        for i, c in enumerate(hand[:self.hand_slots]):
            hand_feats[i] = card_feature_vector(c, owner_card_count=owned)

        pile_feats = np.zeros((PILE_SLOTS, FEATURE_DIM + 1), dtype=np.float32)
        for i, pile in enumerate(self.slot_assignment):
            if pile is not None:
                pile_feats[i] = pile_feature_vector(pile, owner_card_count=owned)

        ps = self._prev_player_state
        player_state_vec = np.array([
            ps.deck_strength,
            ps.score_gap,
            ps.estimated_turns_remaining,
            float(list(Priority).index(ps.current_priority)),
        ], dtype=np.float32)

        obs = {"hand": hand_feats, "piles": pile_feats, "player_state": player_state_vec}
        mask = self._build_mask(request)
        return obs, mask

    def _build_mask(self, request):
        if request.kind == "action":
            return hand_slot_mask(self.hero.hand.cards, request.candidates, self.hand_slots)
        if request.kind == "buy":
            return pile_slot_mask(self.slot_assignment, request.candidates)
        if request.kind == "pile":
            return pile_slot_mask(self.slot_assignment, request.candidates)
        if request.kind == "cards":
            must_exact = request.count if (request.count is not None and request.count >= 0) else None
            slot_mask, _ = multi_select_mask(self.hero.hand.cards, request.candidates, self.hand_slots,
                                              must_exact_count=must_exact)
            return slot_mask
        raise ValueError(request.kind)

    def _decode_action(self, request, action):
        if request.kind == "action":
            return decode_hand_choice(action, self.hero.hand.cards)
        if request.kind in ("buy", "pile"):
            return decode_pile_choice(action, self.slot_assignment)
        if request.kind == "cards":
            indices = [i for i, bit in enumerate(action) if bit] if hasattr(action, "__iter__") else list(action)
            return decode_multi_select(indices, self.hero.hand.cards)
        raise ValueError(request.kind)

    def _terminal_obs(self):
        return {
            "hand": np.zeros((1, FEATURE_DIM), dtype=np.float32),
            "piles": np.zeros((PILE_SLOTS, FEATURE_DIM + 1), dtype=np.float32),
            "player_state": np.zeros(4, dtype=np.float32),
        }
