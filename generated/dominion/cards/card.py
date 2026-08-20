from .cardtypekind import CardTypeKind
from .treasurefacet import TreasureFacet
from .victoryfacet import VictoryFacet

class Card:

	def __init__(self):
		self.name: str = ''
		self.cost: int = 0
		self.types: CardTypeKind = []
		self.treasureFacet = TreasureFacet()
		self.victoryFacet = VictoryFacet()

	def describe(self):
		return f"{self.name} (cost {self.cost})"

	def validate(self):
		if CardTypeKind.Treasure in self.types:
		    assert self.treasureFacet is not None, "Treasure cards must have a treasureFacet"
		if CardTypeKind.Victory in self.types or CardTypeKind.Curse in self.types:
		    assert self.victoryFacet is not None, "Victory/Curse cards must have a victoryFacet"

	def clone(self):
		c = Card()
		c.name = self.name
		c.cost = self.cost
		c.types = list(self.types)
		c.treasureFacet.coinValue = self.treasureFacet.coinValue
		c.victoryFacet.victoryPoints = self.victoryFacet.victoryPoints
		return c

