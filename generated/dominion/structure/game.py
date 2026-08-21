from .player import Player
from .supply import Supply
from ..zones.trash import Trash
from ..expansions.baseexpansion import BaseExpansion
from .supplypile import SupplyPile
from ..zones.deck import Deck
from ..zones.hand import Hand
from ..zones.discardpile import DiscardPile
from ..zones.playarea import PlayArea

class Game:

	def __init__(self):
		self.players = [Player(), Player(), Player(), Player(), Player()]
		self.supply = Supply()
		self.trash = Trash()
		self.expansion = BaseExpansion()

	def setup(self, num_players):
		import random
		
		basic_cards = self.expansion.basic_cards()
		copper = basic_cards["Copper"]
		silver = basic_cards["Silver"]
		gold = basic_cards["Gold"]
		estate = basic_cards["Estate"]
		duchy = basic_cards["Duchy"]
		province = basic_cards["Province"]
		curse = basic_cards["Curse"]
		
		self.supply = Supply()
		self.supply.piles = []
		counts = [(copper, 60 - 7 * num_players), (silver, 40), (gold, 30),
		          (estate, 8 if num_players <= 2 else 12), (duchy, 8 if num_players <= 2 else 12),
		          (province, 8 if num_players <= 2 else 12), (curse, 10 * (num_players - 1))]
		for card_type, count in counts:
		    pile = SupplyPile()
		    pile.cardType = card_type
		    pile.count = count
		    self.supply.piles.append(pile)
		
		all_kingdom_cards = self.expansion.all_kingdom_cards()
		
		# Optional, duck-typed like choose_cards_fn/choose_pile_fn -- not a
		# formal parameter, so setup(num_players)'s existing call sites never
		# need to change. Set game.kingdom_card_names before calling setup()
		# to force a specific 10-card kingdom (e.g. one of BaseExpansion's
		# recommended presets); leave it unset for a random 10-of-26 draw.
		kingdom_card_names = getattr(self, "kingdom_card_names", None)
		if kingdom_card_names is not None:
		    by_name = {c.name: c for c in all_kingdom_cards}
		    chosen_cards = [by_name[n] for n in kingdom_card_names]
		else:
		    chosen_cards = random.sample(all_kingdom_cards, 10)
		
		kingdom_counts = [(card_type, 10) for card_type in chosen_cards]
		for card_type, count in kingdom_counts:
		    pile = SupplyPile()
		    pile.cardType = card_type
		    pile.count = count
		    self.supply.piles.append(pile)
		
		self.trash = Trash()
		self.trash.cards = []
		
		self.players = []
		for i in range(num_players):
		    p = Player()
		    p.deck = Deck()
		    p.deck.cards = []
		    p.hand = Hand()
		    p.hand.cards = []
		    p.discardPile = DiscardPile()
		    p.discardPile.cards = []
		    p.playArea = PlayArea()
		    p.playArea.cards = []
		    p.actions = 1
		    p.buys = 1
		    p.coins = 0
		    for _ in range(7):
		        p.deck.cards.append(copper.clone())
		    for _ in range(3):
		        p.deck.cards.append(estate.clone())
		    random.shuffle(p.deck.cards)
		    p.end_turn()
		    self.players.append(p)

	def is_over(self):
		for pile in self.supply.piles:
		    if pile.cardType.name == "Province" and pile.count <= 0:
		        return True
		empty_piles = sum(1 for pile in self.supply.piles if pile.count <= 0)
		return empty_piles >= 3

