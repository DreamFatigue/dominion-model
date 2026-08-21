"""Interactive test mode: Player 0 starts with one copy of every card in the
game (all 26 Kingdom cards + all 7 basic cards) in hand, a second copy of
each queued up in the draw pile so re-draws can be tested too, and
effectively unlimited actions/buys so nothing runs out mid-test. Player 1 is
a heuristic AI opponent, present so Attack cards (Militia/Witch/Bureaucrat/
Bandit) have a real target to resolve against.

This is a debug harness, not a real game -- it does not aim for a fair or
winnable setup, just maximum surface area to exercise every card's effect.
"""

import random

import questionary

from dominion.structure.game import Game
from dominion.rl.agents import set_heuristic_baseline
from dominion.rl.state import capture_initial_pile_counts

from demo import play_turn
from play_human import human_choose_cards, human_choose_pile, human_choose_action, human_turn

UNLIMITED = 999


def build_test_hand(human, all_card_types):
    human.hand.cards = []
    human.deck.cards = []
    human.discardPile.cards = []
    human.playArea.cards = []

    for card_type in all_card_types:
        human.hand.cards.append(card_type.clone())
    second_copies = [card_type.clone() for card_type in all_card_types]
    random.shuffle(second_copies)
    human.deck.cards = second_copies

    human.actions = UNLIMITED
    human.buys = UNLIMITED
    human.coins = 0


def main():
    game = Game()
    game.kingdom_card_names = game.expansion.kingdom_card_names()  # all 26, not just 10
    game.setup(2)
    human, ai = game.players[0], game.players[1]

    human.choose_cards_fn = human_choose_cards
    human.choose_pile_fn = human_choose_pile
    human.choose_action_fn = human_choose_action
    set_heuristic_baseline(ai)

    all_card_types = list(game.expansion.basic_cards().values()) + game.expansion.all_kingdom_cards()
    build_test_hand(human, all_card_types)

    print("=== TEST MODE: every card in hand, a 2nd copy queued in the draw pile, unlimited actions/buys ===")
    print(f"Your hand ({len(human.hand.cards)} cards): {[c.name for c in human.hand.cards]}")

    initial_pile_counts = capture_initial_pile_counts(game.supply)
    round_num = 0
    while not game.is_over():
        round_num += 1
        print(f"\n========== Round {round_num} ==========")
        human_turn(human, 0, game)
        # Testing mode: refill actions/buys every round instead of the
        # normal 1/1 reset from end_turn(), so a long test session never
        # runs dry.
        human.actions = UNLIMITED
        human.buys = UNLIMITED
        if game.is_over():
            break
        print("\n  --- AI (Player 1)'s turn ---")
        play_turn(ai, 1, game, round_num, initial_pile_counts)

        if not questionary.confirm("Continue testing?", default=True).ask():
            break

    print("\n=== TEST SESSION ENDED ===")


if __name__ == "__main__":
    main()
