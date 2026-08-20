import questionary

from dominion.structure.game import Game
from dominion.cards.cardtypekind import CardTypeKind

from demo import snapshot, report_diff, CARD_DESCRIPTIONS, play_turn
from dominion.rl.state import capture_initial_pile_counts

STOP = object()


def card_choice(c):
    return questionary.Choice(f"{c.name} (cost {c.cost})", value=c)


def pile_choice(p):
    return questionary.Choice(f"{p.cardType.name} (cost {p.cardType.cost}, {p.count} left)", value=p)


def human_choose_cards(candidates, prompt, context, count):
    if not candidates:
        return []
    if count == 1:
        choices = [card_choice(c) for c in candidates]
        choices.append(questionary.Choice("-- Skip --", value=None))
        chosen = questionary.select(prompt, choices=choices).ask()
        return [chosen] if chosen is not None else []
    choices = [card_choice(c) for c in candidates]
    chosen = questionary.checkbox(prompt, choices=choices).ask()
    return chosen or []


def human_choose_pile(candidates, prompt, context):
    if not candidates:
        print(f"    {prompt} -- no eligible options.")
        return None
    choices = [pile_choice(p) for p in candidates]
    choices.append(questionary.Choice("-- Skip --", value=None))
    return questionary.select(prompt, choices=choices).ask()


def human_turn(p, idx, game):
    print(f"\n  === YOUR TURN (Player {idx}) ===")
    print(f"    Hand: {[c.name for c in p.hand.cards]}")
    print(f"    Actions: {p.actions}  Buys: {p.buys}  Coins: {p.coins}")

    # Action phase
    while p.actions > 0:
        action_cards = [c for c in p.hand.cards if CardTypeKind.Action in c.types]
        if not action_cards:
            print("    No action cards in hand.")
            break
        choices = [card_choice(c) for c in action_cards]
        choices.append(questionary.Choice("-- Done with actions --", value=STOP))
        card = questionary.select("Play which action card?", choices=choices).ask()
        if card is STOP or card is None:
            break
        description = CARD_DESCRIPTIONS.get(card.name, "")
        before = snapshot(game)
        p.play_action_card(card, game)
        after = snapshot(game)
        print(f"    Played {card.name}: {description} (actions left: {p.actions})")
        report_diff(before, after, game)
        print(f"    Hand: {[c.name for c in p.hand.cards]}")

    # Buy phase
    if questionary.confirm("Play all treasures in hand?", default=True).ask():
        n_treasures = p.play_treasures()
        print(f"    Played {n_treasures} treasures. Coins: {p.coins}")

    while p.buys > 0:
        affordable = [pile for pile in game.supply.piles
                      if pile.count > 0 and pile.cardType.cost <= p.coins]
        if not affordable:
            print("    Nothing affordable.")
            break
        choices = [pile_choice(pile) for pile in affordable]
        choices.append(questionary.Choice("-- Done buying --", value=STOP))
        pile = questionary.select(f"Buys left: {p.buys}  Coins: {p.coins}  Buy which pile?", choices=choices).ask()
        if pile is STOP or pile is None:
            break
        p.buy_card(pile)
        print(f"    Bought {pile.cardType.name}. Coins left: {p.coins}, buys left: {p.buys}")

    # Cleanup phase
    p.end_turn()
    print(f"    [Cleanup] New hand: {[c.name for c in p.hand.cards]}")


def main():
    game = Game()
    game.setup(2)
    human, ai = game.players[0], game.players[1]

    human.choose_cards_fn = human_choose_cards
    human.choose_pile_fn = human_choose_pile
    ai.choose_cards_fn = ai.ai_choose_cards
    ai.choose_pile_fn = ai.ai_choose_pile
    ai.choose_action_fn = ai.ai_choose_action
    ai.choose_buy_fn = ai.ai_choose_buy

    print("=== Welcome to Dominion! You are Player 0, playing against the AI (Player 1). ===")

    province_pile = next(pile for pile in game.supply.piles if pile.cardType.name == "Province")
    province_answer = questionary.text(
        f"How many Provinces should the supply start with? (default {province_pile.count})",
        default=str(province_pile.count),
        validate=lambda text: text.isdigit() and int(text) > 0 or "Enter a positive whole number",
    ).ask()
    province_pile.count = int(province_answer)
    print(f"Province pile set to {province_pile.count}.")

    visibility = questionary.select(
        "How do you want to play?",
        choices=[
            questionary.Choice("Closed hand -- hide the AI's hand", value="closed"),
            questionary.Choice("Open hand -- show the AI's hand", value="open"),
        ],
    ).ask()
    hidden = {1} if visibility == "closed" else set()

    print(f"Your starting hand: {[c.name for c in human.hand.cards]}")

    initial_pile_counts = capture_initial_pile_counts(game.supply)
    round_num = 0
    while not game.is_over():
        round_num += 1
        print(f"\n========== Round {round_num} ==========")
        human_turn(human, 0, game)
        if game.is_over():
            break
        print("\n  --- AI (Player 1)'s turn ---")
        play_turn(ai, 1, game, round_num, initial_pile_counts, hidden=hidden)

    print("\n=== GAME OVER ===")
    human_score = human.calculate_score()
    ai_score = ai.calculate_score()
    print(f"Your score: {human_score}")
    print(f"AI score: {ai_score}")
    if human_score > ai_score:
        print("You win!")
    elif ai_score > human_score:
        print("AI wins!")
    else:
        print("Tie game!")


if __name__ == "__main__":
    main()
