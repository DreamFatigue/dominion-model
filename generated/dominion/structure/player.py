from ..zones.deck import Deck
from ..zones.hand import Hand
from ..zones.discardpile import DiscardPile
from ..zones.playarea import PlayArea
from ..cards.card import Card
from ..cards.cardtypekind import CardTypeKind
from .supplypile import SupplyPile

class Player:

	def __init__(self):
		self.actions: int = 0
		self.buys: int = 0
		self.coins: int = 0
		self.silverPlayedThisTurn: bool = False
		self.declineMoatReveal: bool = False
		self.deck = Deck()
		self.hand = Hand()
		self.discardPile = DiscardPile()
		self.playArea = PlayArea()

	def draw(self, n):
		import random
		for _ in range(n):
			if not self.deck.cards:
				if not self.discardPile.cards:
					return
				self.deck.cards = self.discardPile.cards
				random.shuffle(self.deck.cards)
				self.discardPile.cards = []
			if self.deck.cards:
				self.hand.cards.append(self.deck.cards.pop())

	def ai_choose_cards(self, candidates, prompt, context, count):
		if context == "cellar_discard":
			return [c for c in candidates if CardTypeKind.Victory in c.types or CardTypeKind.Curse in c.types]
		if context in ("mine_trash", "remodel_trash"):
			return [min(candidates, key=lambda c: c.cost)] if candidates else []
		if context == "militia_discard":
			sorted_candidates = sorted(candidates, key=lambda c: c.cost)
			return sorted_candidates[:count]
		return []

	def ai_choose_pile(self, candidates, prompt, context):
		return max(candidates, key=lambda p: p.cardType.cost) if candidates else None

	def play_action_card(self, card, game):
		if self.actions <= 0:
			return False
		if card not in self.hand.cards:
			return False
		self.actions -= 1
		self.hand.cards.remove(card)
		self.playArea.cards.append(card)

		name = card.name

		if name == "Village":
			self.actions += 2
			self.draw(1)
		elif name == "Smithy":
			self.draw(3)
		elif name == "Market":
			self.actions += 1
			self.buys += 1
			self.coins += 1
			self.draw(1)
		elif name == "Workshop":
			candidates = game.supply.eligible_piles(4, None)
			gain_pile = self.choose_pile_fn(candidates, "Workshop: choose a card to gain (to discard)", "workshop_gain")
			if gain_pile is not None:
				gained = game.supply.take_from_pile(gain_pile)
				self.discardPile.cards.append(gained)
		elif name == "Moat":
			self.draw(2)
		elif name == "Cellar":
			self.actions += 1
			discard_candidates = self.choose_cards_fn(list(self.hand.cards),
			                                           "Cellar: choose any number of cards to discard", "cellar_discard", -1)
			for c in discard_candidates:
				self.hand.cards.remove(c)
				self.discardPile.cards.append(c)
			self.draw(len(discard_candidates))
		elif name == "Merchant":
			self.actions += 1
			self.draw(1)
		elif name == "Militia":
			self.coins += 2
			for other in game.players:
				if other is self:
					continue
				if any(c.name == "Moat" for c in other.hand.cards) and not other.declineMoatReveal:
					continue
				excess = len(other.hand.cards) - 3
				if excess > 0:
					to_discard = other.choose_cards_fn(list(other.hand.cards),
					                                    "Militia: choose " + str(excess) + " card(s) to discard",
					                                    "militia_discard", excess)
					for c in to_discard:
						if c in other.hand.cards:
							other.hand.cards.remove(c)
							other.discardPile.cards.append(c)
		elif name == "Mine":
			treasures_in_hand = [c for c in self.hand.cards if CardTypeKind.Treasure in c.types]
			if treasures_in_hand:
				chosen = self.choose_cards_fn(treasures_in_hand, "Mine: choose a Treasure to trash", "mine_trash", 1)
				if chosen:
					to_trash = chosen[0]
					candidates = game.supply.eligible_piles(to_trash.cost + 3, CardTypeKind.Treasure)
					gain_pile = self.choose_pile_fn(candidates, "Mine: choose a Treasure to gain to hand", "mine_gain")
					if gain_pile is not None:
						gained = game.supply.take_from_pile(gain_pile)
						self.hand.cards.remove(to_trash)
						game.trash.cards.append(to_trash)
						self.hand.cards.append(gained)
		elif name == "Remodel":
			if self.hand.cards:
				chosen = self.choose_cards_fn(list(self.hand.cards), "Remodel: choose a card to trash", "remodel_trash", 1)
				if chosen:
					to_trash = chosen[0]
					candidates = game.supply.eligible_piles(to_trash.cost + 2, None)
					gain_pile = self.choose_pile_fn(candidates, "Remodel: choose a card to gain (to discard)", "remodel_gain")
					if gain_pile is not None:
						gained = game.supply.take_from_pile(gain_pile)
						self.hand.cards.remove(to_trash)
						game.trash.cards.append(to_trash)
						self.discardPile.cards.append(gained)

		return True

	def play_treasures(self):
		treasures = [c for c in self.hand.cards if CardTypeKind.Treasure in c.types]
		merchants_in_play = sum(1 for c in self.playArea.cards if c.name == "Merchant")
		for card in treasures:
			if card.treasureFacet is not None:
				self.coins += card.treasureFacet.coinValue
			if card.name == "Silver" and not self.silverPlayedThisTurn:
				self.coins += merchants_in_play
				self.silverPlayedThisTurn = True
			self.hand.cards.remove(card)
			self.playArea.cards.append(card)
		return len(treasures)

	def buy_card(self, pile):
		if self.buys <= 0:
			return False
		if pile.count <= 0:
			return False
		if self.coins < pile.cardType.cost:
			return False
		self.buys -= 1
		self.coins -= pile.cardType.cost
		pile.count -= 1
		self.discardPile.cards.append(pile.cardType.clone())
		return True

	def end_turn(self):
		self.discardPile.cards.extend(self.hand.cards)
		self.hand.cards = []
		self.discardPile.cards.extend(self.playArea.cards)
		self.playArea.cards = []
		self.draw(5)
		self.actions = 1
		self.buys = 1
		self.coins = 0
		self.silverPlayedThisTurn = False

	def calculate_score(self):
		total = 0
		for zone in (self.deck.cards, self.hand.cards, self.discardPile.cards, self.playArea.cards):
			for c in zone:
				if CardTypeKind.Victory in c.types or CardTypeKind.Curse in c.types:
					total += c.victoryFacet.victoryPoints
		return total
