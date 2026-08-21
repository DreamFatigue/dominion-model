"""Section 3: Action Space.

Per-decision-point stepping: one RL step() = one atomic decision (play one
card, buy one pile), not a whole turn. Treasure-playing is automatic, not a
decision point, so it has no action space here.
"""

import random
import numpy as np

PILE_SLOTS = 17  # 7 basic + 10 Kingdom, per the plan.
HAND_SLOTS_MAX = 60  # fixed padding width for training obs/masks -- see env.py


def compute_hand_slots(supply, cap=HAND_SLOTS_MAX):
    """Generous fixed cap derived from total supply at setup; unused slots
    are simply masked out. Not meant to be tight, just safely unreachable."""
    return min(sum(p.count for p in supply.piles), cap)


def assign_pile_slots(supply, rng=None):
    """Random, fixed-for-the-episode slot -> pile mapping. Randomizing this
    per episode (rather than a canonical/sorted order) is what stops the
    policy from overfitting to "slot 3 is always Village" and is what makes
    random-kingdom training actually work (Section 3)."""
    rng = rng if rng is not None else random
    piles = list(supply.piles)
    rng.shuffle(piles)
    slots = piles[:PILE_SLOTS] + [None] * max(0, PILE_SLOTS - len(piles))
    return slots


def pile_slot_mask(slot_assignment, eligible_piles, include_stop=True):
    """categorical(PILE_SLOTS[+1]) mask for choose_pile_fn / choose_buy_fn.
    `eligible_piles` is the already-filtered candidate list (e.g.
    Supply.eligible_piles(), or the buy-phase affordability filter)."""
    eligible_ids = {id(p) for p in eligible_piles}
    mask = np.array([slot is not None and id(slot) in eligible_ids for slot in slot_assignment], dtype=bool)
    if include_stop:
        mask = np.append(mask, True)  # STOP always legal, per Section 3's table.
    return mask


def decode_pile_choice(action_index, slot_assignment):
    """Returns the chosen SupplyPile, or None for STOP."""
    if action_index >= len(slot_assignment):
        return None
    return slot_assignment[action_index]


def encode_pile_choice(pile, slot_assignment):
    if pile is None:
        return len(slot_assignment)
    for i, slot in enumerate(slot_assignment):
        if slot is pile:
            return i
    raise ValueError(f"pile {pile.cardType.name if pile else pile} not present in slot assignment")


def hand_slot_mask(hand, candidates, hand_slots, include_stop=True):
    """categorical(HAND_SLOTS[+1]) mask for choose_action_fn. `hand` defines
    slot -> card for this decision point; `candidates` are the legal picks
    (e.g. playable Action cards, actions > 0 already checked by caller)."""
    candidate_ids = {id(c) for c in candidates}
    mask = np.zeros(hand_slots, dtype=bool)
    for i, card in enumerate(hand):
        if i >= hand_slots:
            break
        if id(card) in candidate_ids:
            mask[i] = True
    if include_stop:
        mask = np.append(mask, True)
    return mask


def decode_hand_choice(action_index, hand):
    if action_index >= len(hand):
        return None
    return hand[action_index]


def encode_hand_choice(card, hand):
    if card is None:
        return len(hand)
    for i, c in enumerate(hand):
        if c is card:
            return i
    raise ValueError(f"card {card.name if card else card} not present in hand")


def multi_select_mask(hand, candidates, hand_slots, must_exact_count=None):
    """multi-binary(HAND_SLOTS) + commit for choose_cards_fn.

    May vs. Must (Section 3): May (must_exact_count=None, e.g. Cellar) makes
    commit always legal, including empty selection. Must (e.g. Mine/Remodel:
    exactly 1; Militia: exactly `count`) masks commit illegal until exactly
    `must_exact_count` slots are toggled on -- driven dynamically by the
    real `count` already in choose_cards_fn's signature, not a hardcoded
    property of the space.

    Returns (slot_mask, commit_legal_fn): slot_mask marks which slots are
    toggleable at all; commit_legal_fn(n_selected) -> bool.
    """
    candidate_ids = {id(c) for c in candidates}
    slot_mask = np.zeros(hand_slots, dtype=bool)
    for i, card in enumerate(hand):
        if i >= hand_slots:
            break
        if id(card) in candidate_ids:
            slot_mask[i] = True

    if must_exact_count is None:
        def commit_legal_fn(n_selected):
            return True
    else:
        def commit_legal_fn(n_selected):
            return n_selected == must_exact_count

    return slot_mask, commit_legal_fn


def decode_multi_select(action_indices, hand):
    return [hand[i] for i in action_indices if i < len(hand)]


def encode_multi_select(chosen_cards, hand):
    id_to_index = {id(c): i for i, c in enumerate(hand)}
    return [id_to_index[id(c)] for c in chosen_cards if id(c) in id_to_index]
