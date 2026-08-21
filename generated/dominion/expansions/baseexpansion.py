from ..cards.card import Card
from ..cards.cardtypekind import CardTypeKind
from ..cards.treasurefacet import TreasureFacet
from ..cards.victoryfacet import VictoryFacet

class BaseExpansion:

	def basic_cards(self):
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
		
		return {
		    "Copper": make_treasure("Copper", 0, 1),
		    "Silver": make_treasure("Silver", 3, 2),
		    "Gold": make_treasure("Gold", 6, 3),
		    "Estate": make_victory("Estate", 2, 1),
		    "Duchy": make_victory("Duchy", 5, 3),
		    "Province": make_victory("Province", 8, 6),
		    "Curse": make_curse(),
		}

	def all_kingdom_cards(self):
		def make_action(name, cost, extra_types=None):
		    c = Card()
		    c.name = name
		    c.cost = cost
		    c.types = [CardTypeKind.Action] + (extra_types or [])
		    return c
		
		def make_victory(name, cost, points):
		    c = Card()
		    c.name = name
		    c.cost = cost
		    c.types = [CardTypeKind.Victory]
		    c.victoryFacet = VictoryFacet()
		    c.victoryFacet.victoryPoints = points
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
		poacher = make_action("Poacher", 4)
		festival = make_action("Festival", 5)
		laboratory = make_action("Laboratory", 5)
		council_room = make_action("Council Room", 5)
		moneylender = make_action("Moneylender", 4)
		chapel = make_action("Chapel", 2)
		artisan = make_action("Artisan", 6)
		library = make_action("Library", 5)
		harbinger = make_action("Harbinger", 3)
		vassal = make_action("Vassal", 3)
		sentry = make_action("Sentry", 5)
		witch = make_action("Witch", 5, [CardTypeKind.Attack])
		bureaucrat = make_action("Bureaucrat", 4, [CardTypeKind.Attack])
		bandit = make_action("Bandit", 5, [CardTypeKind.Attack])
		throne_room = make_action("Throne Room", 4)
		# Gardens' real VP is computed dynamically from deck size (see
		# dominion.rl.card_constants.COMPUTED_VALUE_VICTORY_CARDS) -- the
		# static victoryPoints=0 here is just a placeholder, never read.
		gardens = make_victory("Gardens", 4, 0)
		
		return [village, smithy, market, workshop, moat, cellar, merchant, militia, mine, remodel,
		        poacher, festival, laboratory, council_room, moneylender, chapel, artisan, library,
		        harbinger, vassal, sentry, witch, bureaucrat, bandit, throne_room, gardens]

	def kingdom_card_names(self):
		return [c.name for c in self.all_kingdom_cards()]

	def recommended_kingdoms(self):
		return {
		    "First Game": ["Cellar", "Market", "Merchant", "Militia", "Mine", "Moat", "Remodel", "Smithy", "Village", "Workshop"],
		    "Size Distortion": ["Artisan", "Bandit", "Bureaucrat", "Chapel", "Festival", "Gardens", "Sentry", "Throne Room", "Witch", "Workshop"],
		    "Deck Top": ["Artisan", "Bureaucrat", "Council Room", "Festival", "Harbinger", "Laboratory", "Moneylender", "Sentry", "Vassal", "Village"],
		    "Sleight of Hand": ["Cellar", "Council Room", "Festival", "Gardens", "Harbinger", "Library", "Militia", "Poacher", "Smithy", "Throne Room"],
		    "Improvements": ["Artisan", "Cellar", "Market", "Merchant", "Mine", "Moat", "Moneylender", "Poacher", "Remodel", "Witch"],
		    "Silver & Gold": ["Bandit", "Bureaucrat", "Chapel", "Harbinger", "Laboratory", "Merchant", "Mine", "Moneylender", "Throne Room", "Vassal"],
		}

