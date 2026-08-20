from .supplypile import SupplyPile
from ..cards.cardtypekind import CardTypeKind

class Supply:

	def __init__(self):
		self.piles = []

	def gain_card_up_to(self, max_cost, required_type):
		candidates = [p for p in self.piles if p.count > 0 and p.cardType.cost <= max_cost]
		if required_type is not None:
			candidates = [p for p in candidates if required_type in p.cardType.types]
		if not candidates:
			return None
		best = max(candidates, key=lambda p: p.cardType.cost)
		best.count -= 1
		return best.cardType.clone()

	def eligible_piles(self, max_cost, required_type):
		candidates = [p for p in self.piles if p.count > 0 and p.cardType.cost <= max_cost]
		if required_type is not None:
			candidates = [p for p in candidates if required_type in p.cardType.types]
		return candidates

	def take_from_pile(self, pile):
		pile.count -= 1
		return pile.cardType.clone()
