"""Manual smoke test for the 16 new Kingdom cards -- exercises every new
elif branch in player.py across two custom kingdoms, using the heuristic
baseline (declines on unknown contexts, which is fine -- we're checking for
crashes, not perfect play)."""

from dominion.structure.game import Game
from dominion.cards.cardtypekind import CardTypeKind
from dominion.rl.agents import set_heuristic_baseline

KINGDOM_A = ["Poacher", "Festival", "Laboratory", "Council Room", "Moneylender",
             "Chapel", "Artisan", "Library", "Harbinger", "Vassal"]
KINGDOM_B = ["Sentry", "Witch", "Bureaucrat", "Bandit", "Throne Room",
             "Gardens", "Village", "Moat", "Smithy", "Cellar"]


def play_turn(p, game):
    while p.actions > 0:
        action_cards = [c for c in p.hand.cards if CardTypeKind.Action in c.types]
        if not action_cards:
            break
        card = p.choose_action_fn(action_cards, "", "action_phase")
        if card is None:
            break
        p.play_action_card(card, game)
    p.play_treasures()
    while p.buys > 0:
        affordable = [pl for pl in game.supply.piles if pl.count > 0 and pl.cardType.cost <= p.coins]
        if not affordable:
            break
        best = p.choose_buy_fn(affordable, "", "buy_phase")
        if best is None:
            break
        p.buy_card(best)
    p.end_turn()


def run_kingdom(name, kingdom_names, rounds=60):
    g = Game()
    g.kingdom_card_names = kingdom_names
    g.setup(2)
    for p in g.players:
        set_heuristic_baseline(p)
    actual_kingdom = sorted(pile.cardType.name for pile in g.supply.piles
                             if pile.cardType.name not in ("Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"))
    assert actual_kingdom == sorted(kingdom_names), f"kingdom mismatch: {actual_kingdom}"
    for round_num in range(1, rounds + 1):
        for p in g.players:
            play_turn(p, g)
        if g.is_over():
            break
    print(f"{name}: rounds={round_num} game_over={g.is_over()} scores={[p.calculate_score() for p in g.players]}")


def run_random_kingdoms(n=5):
    for i in range(n):
        g = Game()
        g.setup(2)
        for p in g.players:
            set_heuristic_baseline(p)
        kingdom = [pile.cardType.name for pile in g.supply.piles
                   if pile.cardType.name not in ("Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse")]
        assert len(kingdom) == 10 and len(set(kingdom)) == 10
        for round_num in range(1, 61):
            for p in g.players:
                play_turn(p, g)
            if g.is_over():
                break
        print(f"random#{i}: kingdom={kingdom} rounds={round_num} scores={[p.calculate_score() for p in g.players]}")


if __name__ == "__main__":
    run_kingdom("KINGDOM_A", KINGDOM_A)
    run_kingdom("KINGDOM_B", KINGDOM_B)
    run_random_kingdoms(5)
    print("NO CRASH")
