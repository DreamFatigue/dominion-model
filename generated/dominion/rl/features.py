"""Section 2: Card Feature Vector.

Every card -- in hand, in a pile, revealed from deck -- is represented by
the same fixed-length feature vector, built from its properties rather
than its identity, so the policy generalizes across random kingdoms.
"""

import numpy as np

from ..cards.cardtypekind import CardTypeKind
from .card_constants import get_constants, COMPUTED_VALUE_VICTORY_CARDS

# Normalization constants. cost/coin_value/victory_points are base-game
# maximums (see Section 2); the rest are maximums across the full 26+16
# planned card set (see Section 7's Feature Vector constants table).
COST_NORM = 8.0
COIN_VALUE_NORM = 3.0
VICTORY_POINTS_NORM = 6.0
ACTIONS_GRANTED_NORM = 2.0
DRAW_GRANTED_NORM = 4.0
BUYS_GRANTED_NORM = 1.0
COINS_GRANTED_NORM = 3.0
CARDS_GAINED_NORM = 1.0
ATTACK_MAGNITUDE_NORM = 3.0
TRASH_FROM_HAND_NORM = 4.0

FEATURE_NAMES = [
    "cost",
    "is_action", "is_treasure", "is_victory", "is_curse", "is_attack", "is_reaction",
    "coin_value",
    "victory_points",
    "actions_granted", "draw_granted", "buys_granted", "coins_granted",
    "cards_gained",
    "attack_magnitude",
    "trash_from_hand", "grants_replacement",
    "has_choice_effect",
]
FEATURE_DIM = len(FEATURE_NAMES)


def _victory_points(card, owner_card_count):
    computed = COMPUTED_VALUE_VICTORY_CARDS.get(card.name)
    if computed is not None:
        if owner_card_count is None:
            return 0
        return computed(owner_card_count)
    return card.victoryFacet.victoryPoints if card.victoryFacet is not None else 0


def card_feature_vector(card, owner_card_count=None):
    """owner_card_count: total cards owned by whoever's evaluating this
    card (deck+hand+discard+playArea), required to score computed-value
    Victory cards (Gardens) correctly. None is safe for non-computed cards.
    """
    c = get_constants(card.name)
    types = card.types
    return np.array([
        card.cost / COST_NORM,
        float(CardTypeKind.Action in types),
        float(CardTypeKind.Treasure in types),
        float(CardTypeKind.Victory in types),
        float(CardTypeKind.Curse in types),
        float(CardTypeKind.Attack in types),
        float(CardTypeKind.Reaction in types),
        (card.treasureFacet.coinValue if card.treasureFacet is not None else 0) / COIN_VALUE_NORM,
        _victory_points(card, owner_card_count) / VICTORY_POINTS_NORM,
        c.actions_granted / ACTIONS_GRANTED_NORM,
        c.draw_granted / DRAW_GRANTED_NORM,
        c.buys_granted / BUYS_GRANTED_NORM,
        c.coins_granted / COINS_GRANTED_NORM,
        c.cards_gained / CARDS_GAINED_NORM,
        c.attack_magnitude / ATTACK_MAGNITUDE_NORM,
        c.trash_from_hand / TRASH_FROM_HAND_NORM,
        float(c.grants_replacement),
        float(c.has_choice_effect),
    ], dtype=np.float32)


def pile_feature_vector(pile, owner_card_count=None):
    """Piles (Supply) get the card feature vector plus a trailing `count`
    (copies remaining), per Section 2."""
    base = card_feature_vector(pile.cardType, owner_card_count)
    return np.concatenate([base, np.array([float(pile.count)], dtype=np.float32)])
