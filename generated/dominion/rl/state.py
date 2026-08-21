"""Section 1: DeckState & PlayerState."""

import random
from dataclasses import dataclass
from enum import Enum

from ..cards.cardtypekind import CardTypeKind
from .card_constants import get_constants, COMPUTED_VALUE_VICTORY_CARDS

# --- Deck Strength weighting (all "start rough, tune during training" per the plan) ---
W = 1.0 / 3.0
W_ATTACK = 0.5
W_DEFENSE = 0.5
W_UPGRADE_MAX = 1.0
W_THIN_MAX = 0.7
W_GAIN_MAX = 1.0
T_REFERENCE = 10.0

# --- Current Priority thresholds (v1 heuristic, tunable) ---
ENDGAME_TURNS_THRESHOLD = 4
GROWTH_GAIN_THRESHOLD = 0.5
DEFENSE_DENSITY_THRESHOLD = 0.05
ACTIONS_STRANDED_THRESHOLD = 0.0
DRAW_RATIO_THRESHOLD = 0.15
MONEY_TARGET = 5.0
ATTACK_LOW_THRESHOLD = 0.0


@dataclass
class DeckState:
    size: int
    avg_card_cost: float
    actions_per_turn: float
    card_draw_per_turn: float
    money_generated_per_turn: float
    buys_granted_per_turn: float
    cards_gained_per_turn: float
    playable_cards_ratio: float
    upgrade_ability: float
    thinning_ability: float
    attack_pressure_generated: float
    defense_density: float
    gain_potential: float


class Priority(Enum):
    VPs = "VPs"
    Growth = "Growth"
    Defense = "Defense"
    Actions = "Actions"
    Draw = "Draw"
    Money = "Money"
    Attack = "Attack"
    Shedding = "Shedding"


@dataclass
class PlayerState:
    estimated_turns_remaining: float
    deck_strength: float
    score_gap: float
    current_priority: Priority
    deck_state: DeckState


@dataclass
class KingdomContext:
    has_growth_card: bool
    has_attack_cards: bool
    has_defense_card: bool


def owned_cards(player):
    return (player.deck.cards + player.hand.cards
            + player.discardPile.cards + player.playArea.cards)


def analyze_kingdom(supply):
    has_growth = any(pile.cardType.name in COMPUTED_VALUE_VICTORY_CARDS for pile in supply.piles)
    has_attack = any(CardTypeKind.Attack in pile.cardType.types for pile in supply.piles)
    has_defense = any(CardTypeKind.Reaction in pile.cardType.types for pile in supply.piles)
    return KingdomContext(has_growth_card=has_growth, has_attack_cards=has_attack, has_defense_card=has_defense)


def _simulate_one_turn(hand, deck_pool):
    """Chain-simulates a single turn given a starting hand and the
    remaining shuffled deck to draw further from. Mutates nothing; returns
    (outputs, leftover_deck_pool) so the caller can chain turns together."""
    actions_remaining = 1
    buys_total = coins_total = draw_total = gained_total = attack_total = 0.0
    played = set()

    while actions_remaining > 0:
        playable = [c for c in hand if id(c) not in played and CardTypeKind.Action in c.types]
        if not playable:
            break
        # Prefer support cards (actions_granted) before terminals, per Section 1.
        card = max(playable, key=lambda c: (
            get_constants(c.name).actions_granted,
            get_constants(c.name).draw_granted,
            get_constants(c.name).coins_granted,
        ))
        cc = get_constants(card.name)
        played.add(id(card))
        actions_remaining += cc.actions_granted - 1
        buys_total += cc.buys_granted
        coins_total += cc.coins_granted
        gained_total += cc.cards_gained
        attack_total += cc.attack_magnitude
        draw_total += cc.draw_granted
        if cc.draw_granted > 0 and deck_pool:
            n = min(cc.draw_granted, len(deck_pool))
            hand = hand + deck_pool[:n]
            deck_pool = deck_pool[n:]

    for c in hand:
        if id(c) not in played and CardTypeKind.Treasure in c.types:
            coins_total += c.treasureFacet.coinValue if c.treasureFacet is not None else 0

    outputs = {
        "actions": actions_remaining - 1,
        "draw": draw_total,
        "money": coins_total,
        "buys": buys_total,
        "gained": gained_total,
        "attack": attack_total,
    }
    return outputs, deck_pool


def _chain_simulation(cards, num_shuffles=3, rng=None):
    """Section 1's chain simulation. Each shuffle deals the *entire* owned
    pool out into sequential 5-card turns (last turn gets whatever's left)
    and simulates straight through -- so every card is used and turn-to-turn
    deck cycling is modeled -- rather than independently resampling just a
    starting hand, which under-covers a large deck for the same compute.
    Averages across a few independent shuffles for variance reduction."""
    totals = {"actions": 0.0, "draw": 0.0, "money": 0.0, "buys": 0.0, "gained": 0.0, "attack": 0.0}
    if not cards:
        return totals
    rng = rng if rng is not None else random

    turn_count = 0
    for _ in range(num_shuffles):
        deck_pool = list(cards)
        rng.shuffle(deck_pool)
        while deck_pool:
            hand_size = min(5, len(deck_pool))
            hand, deck_pool = deck_pool[:hand_size], deck_pool[hand_size:]
            outputs, deck_pool = _simulate_one_turn(hand, deck_pool)
            for k, v in outputs.items():
                totals[k] += v
            turn_count += 1

    n = float(turn_count) if turn_count else 1.0
    return {k: v / n for k, v in totals.items()}


def compute_deck_state(player, num_shuffles=3, rng=None):
    cards = owned_cards(player)
    size = len(cards)
    if size == 0:
        return DeckState(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    avg_card_cost = sum(c.cost for c in cards) / size
    sim = _chain_simulation(cards, num_shuffles=num_shuffles, rng=rng)

    playable_count = sum(1 for c in cards
                          if CardTypeKind.Action in c.types or CardTypeKind.Treasure in c.types)
    playable_cards_ratio = playable_count / size

    upgrade_trash = sum(get_constants(c.name).trash_from_hand for c in cards
                         if get_constants(c.name).grants_replacement == 1)
    thin_trash = sum(get_constants(c.name).trash_from_hand for c in cards
                      if get_constants(c.name).grants_replacement == 0
                      and get_constants(c.name).trash_from_hand > 0)
    upgrade_ability = (upgrade_trash / size) * (1 - playable_cards_ratio)
    thinning_ability = (thin_trash / size) * (1 - playable_cards_ratio)

    defense_density = sum(1 for c in cards if CardTypeKind.Reaction in c.types) / size
    gain_potential = sim["buys"] + sim["gained"]

    return DeckState(
        size=size,
        avg_card_cost=avg_card_cost,
        actions_per_turn=sim["actions"],
        card_draw_per_turn=sim["draw"],
        money_generated_per_turn=sim["money"],
        buys_granted_per_turn=sim["buys"],
        cards_gained_per_turn=sim["gained"],
        playable_cards_ratio=playable_cards_ratio,
        upgrade_ability=upgrade_ability,
        thinning_ability=thinning_ability,
        attack_pressure_generated=sim["attack"],
        defense_density=defense_density,
        gain_potential=gain_potential,
    )


def capture_initial_pile_counts(supply):
    """Call once right after Game.setup() -- feeds estimate_turns_remaining."""
    return {p.cardType.name: p.count for p in supply.piles}


def estimate_turns_remaining(supply, initial_pile_counts, round_num):
    if round_num <= 0:
        # Turn-1 edge case: no purchase history exists yet to estimate a
        # depletion rate from. A uniform "1 purchase/round/player" prior
        # applied per-pile looked reasonable on paper but, against Province's
        # starting count of 8, produced ~4 turns remaining on turn 1 --
        # nobody buys a Province turn 1, so this fired the VPs priority
        # immediately. Return a large "plenty of turns left" sentinel
        # instead until real purchase data accumulates.
        return T_REFERENCE * 2.0

    def per_pile_estimate(pile):
        initial = initial_pile_counts.get(pile.cardType.name, pile.count)
        consumed = initial - pile.count
        if consumed <= 0:
            # Nobody has bought from this pile yet -- it isn't an imminent
            # depletion threat, so don't fabricate a purchase rate for it
            # (that previously pinned every untouched pile's estimate to a
            # constant, artificially low value for as long as it stayed
            # untouched). Treat as "not a threat yet".
            return float("inf")
        avg_purchases_per_round = consumed / round_num
        return pile.count / avg_purchases_per_round

    province_pile = next((p for p in supply.piles if p.cardType.name == "Province"), None)
    province_estimate = per_pile_estimate(province_pile) if province_pile is not None else float("inf")
    all_estimates = sorted(per_pile_estimate(p) for p in supply.piles)
    three_pile_estimate = all_estimates[2] if len(all_estimates) >= 3 else float("inf")
    return min(province_estimate, three_pile_estimate)


def compute_score_gap(player, all_players):
    opponents = [p for p in all_players if p is not player]
    if not opponents:
        return 0.0
    return player.calculate_score() - max(p.calculate_score() for p in opponents)


def w_upgrade(turns_remaining):
    return W_UPGRADE_MAX * min(1.0, turns_remaining / T_REFERENCE)


def w_thin(turns_remaining):
    return W_THIN_MAX * min(1.0, turns_remaining / T_REFERENCE)


def w_gain(turns_remaining, kingdom_has_growth_card):
    if not kingdom_has_growth_card:
        return 0.0
    return W_GAIN_MAX * min(1.0, turns_remaining / T_REFERENCE)


def compute_deck_strength(deck_state, turns_remaining, kingdom):
    return (
        W * (deck_state.actions_per_turn + deck_state.card_draw_per_turn + deck_state.money_generated_per_turn)
        + W_ATTACK * deck_state.attack_pressure_generated
        + W_DEFENSE * deck_state.defense_density
        + w_upgrade(turns_remaining) * deck_state.upgrade_ability
        + w_thin(turns_remaining) * deck_state.thinning_ability
        + w_gain(turns_remaining, kingdom.has_growth_card) * deck_state.gain_potential
    )


def compute_current_priority(deck_state, turns_remaining, kingdom):
    if turns_remaining <= ENDGAME_TURNS_THRESHOLD:
        return Priority.VPs
    if kingdom.has_growth_card and deck_state.gain_potential > GROWTH_GAIN_THRESHOLD:
        return Priority.Growth
    if (kingdom.has_attack_cards and kingdom.has_defense_card
            and deck_state.defense_density < DEFENSE_DENSITY_THRESHOLD):
        return Priority.Defense
    if deck_state.actions_per_turn <= ACTIONS_STRANDED_THRESHOLD:
        return Priority.Actions
    if deck_state.size > 0 and (deck_state.card_draw_per_turn / deck_state.size) < DRAW_RATIO_THRESHOLD:
        return Priority.Draw
    if deck_state.money_generated_per_turn < MONEY_TARGET:
        return Priority.Money
    if kingdom.has_attack_cards and deck_state.attack_pressure_generated <= ATTACK_LOW_THRESHOLD:
        return Priority.Attack
    return Priority.Shedding


def compute_player_state(player, all_players, supply, initial_pile_counts, round_num,
                          num_shuffles=3, rng=None):
    kingdom = analyze_kingdom(supply)
    turns_remaining = estimate_turns_remaining(supply, initial_pile_counts, round_num)
    deck_state = compute_deck_state(player, num_shuffles=num_shuffles, rng=rng)
    strength = compute_deck_strength(deck_state, turns_remaining, kingdom)
    score_gap = compute_score_gap(player, all_players)
    priority = compute_current_priority(deck_state, turns_remaining, kingdom)
    return PlayerState(
        estimated_turns_remaining=turns_remaining,
        deck_strength=strength,
        score_gap=score_gap,
        current_priority=priority,
        deck_state=deck_state,
    )
