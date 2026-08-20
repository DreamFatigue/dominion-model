from .player import Player
from .supply import Supply
from .supplypile import SupplyPile
from ..zones.trash import Trash
from ..zones.deck import Deck
from ..zones.hand import Hand
from ..zones.discardpile import DiscardPile
from ..zones.playarea import PlayArea
from ..cards.card import Card
from ..cards.cardtypekind import CardTypeKind
from ..cards.treasurefacet import TreasureFacet
from ..cards.victoryfacet import VictoryFacet

class Game:

	def __init__(self):
		self.players = [Player(), Player(), Player(), Player(), Player()]
		self.supply = Supply()
		self.trash = Trash()

	def setup(self, num_players):
		import random
		
		def make_treasure(name, cost, value):
		    c = Card()
		    c.name = name
		    c.cost = cost
		    c.types = [CardTypeKind.Treasure]
		    c.treasureFacet = TreasureFacet()
		    c.treasureFacet.coinValue = value
		    return c
		
		def make_victory(name, cost, points):
		    c = Card()
		    c.name = name
		    c.cost = cost
		    c.types = [CardTypeKind.Victory]
		    c.victoryFacet = VictoryFacet()
		    c.victoryFacet.victoryPoints = points
		    return c
		
		def make_curse():
		    c = Card()
		    c.name = "Curse"
		    c.cost = 0
		    c.types = [CardTypeKind.Curse]
		    c.victoryFacet = VictoryFacet()
		    c.victoryFacet.victoryPoints = -1
		    return c
		
		copper = make_treasure("Copper", 0, 1)
		silver = make_treasure("Silver", 3, 2)
		gold = make_treasure("Gold", 6, 3)
		estate = make_victory("Estate", 2, 1)
		duchy = make_victory("Duchy", 5, 3)
		province = make_victory("Province", 8, 6)
		curse = make_curse()
		
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
		
		def make_action(name, cost, extra_types=None):
		    c = Card()
		    c.name = name
		    c.cost = cost
		    c.types = [CardTypeKind.Action] + (extra_types or [])
		    return c
		
		village = make_action("Village", 3)
		smithy = make_action("Smithy", 4)
		market = make_action("Market", 5)
		workshop = make_action("Workshop", 3)
		moat = make_action("Moat", 2, [CardTypeKind.Reaction])
		cellar = make_action("Cellar", 2)
		merchant = make_action("Merchant", 3)
		militia = make_action("Militia", 4, [CardTypeKind.Attack])
		mine = make_action("Mine", 5)
		remodel = make_action("Remodel", 4)
		
		kingdom_counts = [(village, 10), (smithy, 10), (market, 10), (workshop, 10), (moat, 10),
		                  (cellar, 10), (merchant, 10), (militia, 10), (mine, 10), (remodel, 10)]
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
		        p.deck.cards.append(make_treasure("Copper", 0, 1))
		    for _ in range(3):
		        p.deck.cards.append(make_victory("Estate", 2, 1))
		    random.shuffle(p.deck.cards)
		    p.end_turn()
		    self.players.append(p)

	def is_over(self):
		for pile in self.supply.piles:
		    if pile.cardType.name == "Province" and pile.count <= 0:
		        return True
		empty_piles = sum(1 for pile in self.supply.piles if pile.count <= 0)
		return empty_piles >= 3

