from dominion.structure.game import Game
from dominion.cards.cardtypekind import CardTypeKind
from dominion.rl.state import compute_player_state, capture_initial_pile_counts


CARD_DESCRIPTIONS = {
    "Village": "+1 Card, +2 Actions",
    "Smithy": "+3 Cards",
    "Market": "+1 Card, +1 Action, +1 Buy, +1 Coin",
    "Workshop": "Gain a card costing up to 4",
    "Moat": "+2 Cards (Reaction: blocks Attacks)",
    "Cellar": "+1 Action, discard any number, draw that many",
    "Merchant": "+1 Card, +1 Action (+1 Coin the first Silver played this turn)",
    "Militia": "+2 Coins, other players discard down to 3",
    "Mine": "Trash a Treasure, gain a Treasure costing up to 3 more (to hand)",
    "Remodel": "Trash a card, gain a card costing up to 2 more (to discard)",
    "Poacher": "+1 Card, +1 Action, +1 Coin; discard 1 per empty Supply pile",
    "Festival": "+2 Actions, +1 Buy, +2 Coins",
    "Laboratory": "+2 Cards, +1 Action",
    "Council Room": "+4 Cards, +1 Buy; each other player draws 1",
    "Moneylender": "Trash a Copper from hand for +3 Coins",
    "Chapel": "Trash up to 4 cards from hand",
    "Artisan": "Gain a card costing up to 5 (to hand); put a card from hand onto deck",
    "Library": "Draw until you have 7 cards in hand, may skip drawn Actions",
    "Harbinger": "+1 Card, +1 Action; may put a card from discard onto deck",
    "Vassal": "+2 Coins; discard top of deck, may play it if it's an Action",
    "Sentry": "+1 Card, +1 Action; look at top 2, trash/discard/reorder them",
    "Witch": "+2 Cards; each other player gains a Curse",
    "Bureaucrat": "Gain a Silver to deck; each other player topdecks a Victory card",
    "Bandit": "Gain a Gold; each other player trashes a revealed Treasure",
    "Throne Room": "Play an Action card from hand twice",
    "Gardens": "Victory card worth 1 VP per 10 cards you own, rounded down",
}


def hand_names(player):
    return [c.name for c in player.hand.cards]


def print_hand(player, idx, hidden=None):
    if hidden and idx in hidden:
        print(f"      Player {idx} hand: [hidden] ({len(player.hand.cards)} cards)")
        return
    print(f"      Player {idx} hand: {hand_names(player)}")


def pause():
    input("    [Press Enter to continue] ")


def print_status(player, idx):
    print(f"    Player {idx}: actions={player.actions} buys={player.buys} coins={player.coins} "
          f"hand={len(player.hand.cards)} deck={len(player.deck.cards)} "
          f"discard={len(player.discardPile.cards)} play_area={len(player.playArea.cards)}")


def snapshot(game):
    players_snap = {i: (list(p.hand.cards), list(p.discardPile.cards)) for i, p in enumerate(game.players)}
    trash_snap = list(game.trash.cards)
    all_cards_before = set()
    for p in game.players:
        for zone in (p.hand.cards, p.discardPile.cards, p.deck.cards, p.playArea.cards):
            all_cards_before.update(id(c) for c in zone)
    return players_snap, trash_snap, all_cards_before


def report_diff(before, after, game, hidden=None):
    players_before, trash_before, all_before = before
    players_after, trash_after, _ = after

    trashed = [c for c in trash_after if c not in trash_before]
    if trashed:
        print(f"      Trashed: {[c.name for c in trashed]}")

    for i, p in enumerate(game.players):
        hand_before, discard_before = players_before[i]
        hand_after, discard_after = players_after[i]

        new_in_hand = [c for c in hand_after if c not in hand_before]
        new_in_discard = [c for c in discard_after if c not in discard_before]

        gained_to_hand = [c for c in new_in_hand if id(c) not in all_before]
        gained_to_discard = [c for c in new_in_discard if id(c) not in all_before]
        discarded = [c for c in new_in_discard if id(c) in all_before]

        if gained_to_hand:
            print(f"      Player {i} gained to hand: {[c.name for c in gained_to_hand]}")
        if gained_to_discard:
            print(f"      Player {i} gained: {[c.name for c in gained_to_discard]}")
        if discarded:
            print(f"      Player {i} discarded: {[c.name for c in discarded]}")
        if hand_before != hand_after:
            print_hand(p, i, hidden)


def play_turn(p, idx, game, round_num, initial_pile_counts, hidden=None):
    ps = compute_player_state(p, game.players, game.supply, initial_pile_counts, round_num)
    p.current_priority = ps.current_priority
    priority_label = "[hidden]" if (hidden and idx in hidden) else ps.current_priority.value
    print(f"  --- Player {idx}'s turn --- (priority: {priority_label})")
    print_hand(p, idx, hidden)

    # Action phase
    played_any = False
    while p.actions > 0:
        action_cards = [c for c in p.hand.cards if CardTypeKind.Action in c.types]
        if not action_cards:
            break
        card = p.choose_action_fn(action_cards, "Choose an action card to play", "action_phase")
        if card is None:
            break
        description = CARD_DESCRIPTIONS.get(card.name, "")
        before = snapshot(game)
        p.play_action_card(card, game)
        after = snapshot(game)
        played_any = True
        print(f"    [Action Phase] Played {card.name}: {description} (actions left: {p.actions})")
        if card.name == "Militia":
            for i, other in enumerate(game.players):
                if other is p:
                    continue
                has_moat = any(c.name == "Moat" for c in other.hand.cards)
                if has_moat and not other.declineMoatReveal:
                    print(f"      Player {i} reveals Moat to block the attack!")
                elif has_moat and other.declineMoatReveal:
                    print(f"      Player {i} has Moat but chooses not to reveal it")
        report_diff(before, after, game, hidden)
    if not played_any:
        print("    [Action Phase] No action cards played")
    pause()

    # Buy phase
    n_treasures = p.play_treasures()
    print(f"    [Buy Phase] Played {n_treasures} treasures (coins: {p.coins})")
    bought_any = False
    while p.buys > 0:
        affordable = [pile for pile in game.supply.piles
                      if pile.count > 0 and pile.cardType.cost <= p.coins]
        if not affordable:
            break
        best = p.choose_buy_fn(affordable, "Choose a pile to buy", "buy_phase")
        if best is None:
            break
        p.buy_card(best)
        bought_any = True
        print(f"    [Buy Phase] Bought {best.cardType.name} (coins left: {p.coins}, buys left: {p.buys})")
    if not bought_any:
        print("    [Buy Phase] No cards bought")
    pause()

    # Cleanup phase
    discarded_cleanup = [c.name for c in p.hand.cards] + [c.name for c in p.playArea.cards]
    p.end_turn()
    print("    [Cleanup Phase]")
    print(f"      Discarded: {discarded_cleanup}")
    print_hand(p, idx, hidden)
    print_status(p, idx)
    pause()


def main():
    game = Game()
    game.setup(2)
    for p in game.players:
        p.choose_cards_fn = p.ai_choose_cards
        p.choose_pile_fn = p.ai_choose_pile
        p.choose_action_fn = p.ai_choose_action
        p.choose_buy_fn = p.ai_choose_buy

    print("=== Initial setup ===")
    for i, p in enumerate(game.players):
        print_status(p, i)
        print_hand(p, i)
    print("Supply piles:", [(pile.cardType.name, pile.count) for pile in game.supply.piles])
    print("Game over?", game.is_over())

    initial_pile_counts = capture_initial_pile_counts(game.supply)
    round_num = 0
    while not game.is_over() and round_num < 30:
        round_num += 1
        print(f"\n=== Round {round_num} ===")
        for i, p in enumerate(game.players):
            play_turn(p, i, game, round_num, initial_pile_counts)
            if game.is_over():
                break

    print(f"\n=== Game ended after {round_num} rounds ===")
    print("Game over?", game.is_over())
    for i, p in enumerate(game.players):
        print_status(p, i)
    print("Supply piles:", [(pile.cardType.name, pile.count) for pile in game.supply.piles])
    print("Trash:", [c.name for c in game.trash.cards])


if __name__ == "__main__":
    main()
