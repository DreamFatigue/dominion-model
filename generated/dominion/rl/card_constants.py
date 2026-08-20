"""Per-card constants for fields Section 2 of the RL plan calls out as
hand-tuned, not derivable from the base Card/CardTypeKind/TreasureFacet/
VictoryFacet model. Keyed by Card.name. Cards with no entry (all basic
Treasures/Victory cards, and any Kingdom card not yet implemented) get the
all-zero DEFAULT.

cards_gained is net *new* cards added to the owning player's total count.
Mine/Remodel are net-zero (trash 1, gain 1 back) so they carry
trash_from_hand/grants_replacement instead, never cards_gained -- counting
both would double-count the same card as both a gain and a size-neutral
upgrade.
"""

from collections import namedtuple

CardConstants = namedtuple("CardConstants", [
    "actions_granted",
    "draw_granted",
    "buys_granted",
    "coins_granted",
    "cards_gained",
    "attack_magnitude",
    "trash_from_hand",
    "grants_replacement",
    "has_choice_effect",
])

DEFAULT = CardConstants(0, 0, 0, 0, 0, 0, 0, 0, 0)

CARD_CONSTANTS = {
    "Village": CardConstants(actions_granted=2, draw_granted=1, buys_granted=0,
                              coins_granted=0, cards_gained=0, attack_magnitude=0,
                              trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Smithy": CardConstants(actions_granted=0, draw_granted=3, buys_granted=0,
                             coins_granted=0, cards_gained=0, attack_magnitude=0,
                             trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Market": CardConstants(actions_granted=1, draw_granted=1, buys_granted=1,
                             coins_granted=1, cards_gained=0, attack_magnitude=0,
                             trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Workshop": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                               coins_granted=0, cards_gained=1, attack_magnitude=0,
                               trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Moat": CardConstants(actions_granted=0, draw_granted=2, buys_granted=0,
                           coins_granted=0, cards_gained=0, attack_magnitude=0,
                           trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Cellar": CardConstants(actions_granted=1, draw_granted=0, buys_granted=0,
                             coins_granted=0, cards_gained=0, attack_magnitude=0,
                             trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Merchant": CardConstants(actions_granted=1, draw_granted=1, buys_granted=0,
                               coins_granted=0, cards_gained=0, attack_magnitude=0,
                               trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Militia": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                              coins_granted=2, cards_gained=0, attack_magnitude=2,
                              trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Mine": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                           coins_granted=0, cards_gained=0, attack_magnitude=0,
                           trash_from_hand=1, grants_replacement=1, has_choice_effect=1),
    "Remodel": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                              coins_granted=0, cards_gained=0, attack_magnitude=0,
                              trash_from_hand=1, grants_replacement=1, has_choice_effect=1),

    # --- Section 7's 16 new Kingdom cards ---
    "Poacher": CardConstants(actions_granted=1, draw_granted=1, buys_granted=0,
                              coins_granted=1, cards_gained=0, attack_magnitude=0,
                              trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Festival": CardConstants(actions_granted=2, draw_granted=0, buys_granted=1,
                               coins_granted=2, cards_gained=0, attack_magnitude=0,
                               trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Laboratory": CardConstants(actions_granted=1, draw_granted=2, buys_granted=0,
                                 coins_granted=0, cards_gained=0, attack_magnitude=0,
                                 trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Council Room": CardConstants(actions_granted=0, draw_granted=4, buys_granted=1,
                                   coins_granted=0, cards_gained=0, attack_magnitude=0,
                                   trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Moneylender": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                                  coins_granted=3, cards_gained=0, attack_magnitude=0,
                                  trash_from_hand=1, grants_replacement=0, has_choice_effect=1),
    "Chapel": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                             coins_granted=0, cards_gained=0, attack_magnitude=0,
                             trash_from_hand=4, grants_replacement=0, has_choice_effect=1),
    "Artisan": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                              coins_granted=0, cards_gained=1, attack_magnitude=0,
                              trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Library": CardConstants(actions_granted=0, draw_granted=2, buys_granted=0,
                              coins_granted=0, cards_gained=0, attack_magnitude=0,
                              trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Harbinger": CardConstants(actions_granted=1, draw_granted=1, buys_granted=0,
                                coins_granted=0, cards_gained=0, attack_magnitude=0,
                                trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Vassal": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                             coins_granted=2, cards_gained=0, attack_magnitude=0,
                             trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Sentry": CardConstants(actions_granted=1, draw_granted=1, buys_granted=0,
                             coins_granted=0, cards_gained=0, attack_magnitude=0,
                             trash_from_hand=2, grants_replacement=0, has_choice_effect=1),
    "Witch": CardConstants(actions_granted=0, draw_granted=2, buys_granted=0,
                            coins_granted=0, cards_gained=0, attack_magnitude=3,
                            trash_from_hand=0, grants_replacement=0, has_choice_effect=0),
    "Bureaucrat": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                                 coins_granted=0, cards_gained=1, attack_magnitude=1,
                                 trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Bandit": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                             coins_granted=0, cards_gained=1, attack_magnitude=2,
                             trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    "Throne Room": CardConstants(actions_granted=0, draw_granted=0, buys_granted=0,
                                  coins_granted=0, cards_gained=0, attack_magnitude=0,
                                  trash_from_hand=0, grants_replacement=0, has_choice_effect=1),
    # Gardens (Tier 6) isn't an Action card -- no entry needed, its scoring
    # is handled entirely via COMPUTED_VALUE_VICTORY_CARDS below.
}

# Victory cards whose victoryPoints must be computed from the owning
# player's total owned card count rather than read as a static field.
# See Section 2's "Computed-value cards note" / Section 7 Tier 6.
COMPUTED_VALUE_VICTORY_CARDS = {
    "Gardens": lambda owned_card_count: owned_card_count // 10,
}


def get_constants(card_name):
    return CARD_CONSTANTS.get(card_name, DEFAULT)
