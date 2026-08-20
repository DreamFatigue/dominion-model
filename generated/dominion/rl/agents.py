"""Section 5: baseline opponents (for evaluation, not the self-play pool).

Heuristic baseline is just Player's own ai_choose_* methods -- Section 5
calls for "existing ai_choose_cards/ai_choose_pile logic... plus simple
greedy versions of choose_buy_fn/choose_action_fn to match demo.py style",
which is exactly what those methods already are (see player.py). Random
baseline below samples uniformly among legal actions -- a sanity floor.
"""

import random


def set_heuristic_baseline(player):
    player.choose_cards_fn = player.ai_choose_cards
    player.choose_pile_fn = player.ai_choose_pile
    player.choose_action_fn = player.ai_choose_action
    player.choose_buy_fn = player.ai_choose_buy


def random_choose_cards(candidates, prompt, context, count):
    """count follows the existing convention already used by Cellar/Mine/
    Militia in player.py: -1 = May, any number; >=0 = Must, exactly count."""
    if not candidates:
        return []
    if count is not None and count >= 0:
        k = min(count, len(candidates))
    else:
        k = random.randint(0, len(candidates))
    return random.sample(candidates, k)


def random_choose_pile(candidates, prompt, context):
    options = list(candidates) + [None]
    return random.choice(options)


def random_choose_action(candidates, prompt, context):
    options = list(candidates) + [None]
    return random.choice(options)


def random_choose_buy(candidates, prompt, context):
    options = list(candidates) + [None]
    return random.choice(options)


def set_random_baseline(player):
    player.choose_cards_fn = random_choose_cards
    player.choose_pile_fn = random_choose_pile
    player.choose_action_fn = random_choose_action
    player.choose_buy_fn = random_choose_buy
