from ..zones.deck import Deck
from ..zones.hand import Hand
from ..zones.discardpile import DiscardPile
from ..zones.playarea import PlayArea
from ..cards.card import Card
from .supplypile import SupplyPile
from ..cards.cardtypekind import CardTypeKind
from ..cards.descriptions import CARD_DESCRIPTIONS

class Player:

	def __init__(self):
		self.known_top_cards: Card = []
		self.actions: int = 0
		self.buys: int = 0
		self.coins: int = 0
		self.silverPlayedThisTurn: bool = False
		self.current_priority : None = None
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
				self.known_top_cards = []
			if self.deck.cards:
				drawn = self.deck.cards.pop()
				if self.known_top_cards and self.known_top_cards[0] is drawn:
					self.known_top_cards.pop(0)
				self.hand.cards.append(drawn)

	def _is_junk_to_trash(self, card):
		"""Shared by chapel_trash/sentry_trash. Curses are always junk;
		Estates are junk unless current_priority is Growth (Gardens wants
		total card count, not quality); Coppers are only junk once priority
		has actually shifted to Shedding."""
		from ..rl.state import Priority
		if card.name == "Curse":
			return True
		if card.name == "Estate":
			return self.current_priority != Priority.Growth
		if card.name == "Copper":
			return self.current_priority == Priority.Shedding
		return False

	def ai_choose_cards(self, candidates, prompt, context, count):
		from ..rl.state import Priority
		priority = self.current_priority
		
		if context == "reveal_moat":
			# Revealing Moat is never a bad trade -- it's free and fully
			# blocks the attack -- so the heuristic AI always takes it.
			return list(candidates)
		if context == "cellar_discard":
			return [c for c in candidates if CardTypeKind.Victory in c.types or CardTypeKind.Curse in c.types]
		if context in ("mine_trash", "remodel_trash"):
			return [min(candidates, key=lambda c: c.cost)] if candidates else []
		if context == "militia_discard":
			sorted_candidates = sorted(candidates, key=lambda c: c.cost)
			return sorted_candidates[:count]
		
		if context == "chapel_trash":
			return [c for c in candidates if self._is_junk_to_trash(c)][:4]
		if context == "poacher_discard":
			return sorted(candidates, key=lambda c: c.cost)[:count]
		if context == "moneylender_trash":
			# Always worth it: lose a Copper (worth 1), gain 3 Coins this turn.
			return candidates[:1]
		if context == "artisan_topdeck":
			# Keep the best of hand out this turn; put the worst back to draw again later.
			return [min(candidates, key=lambda c: c.cost)] if candidates else []
		if context == "library_skip":
			# Skip drawn Actions you can't use this turn anyway, or don't
			# currently want (priority has shifted off Actions/Draw).
			if self.actions <= 0 or priority in (Priority.Money, Priority.VPs):
				return list(candidates)
			return []
		if context == "harbinger_topdeck":
			if not candidates:
				return []
			best = max(candidates, key=lambda c: c.cost)
			if best.cost <= 0 and priority != Priority.Money:
				return []
			return [best]
		if context == "vassal_play":
			# A free extra Action play is essentially always worth taking.
			return list(candidates)
		if context == "sentry_trash":
			return [c for c in candidates if self._is_junk_to_trash(c)]
		if context == "sentry_discard":
			return [c for c in candidates if CardTypeKind.Victory in c.types or c.name == "Copper"]
		if context == "sentry_reorder":
			# Keep whatever's left in its current (post-trash/discard) order.
			return list(candidates)
		if context == "bureaucrat_topdeck":
			# Forced choice: minimize the loss by topdecking the cheapest VP.
			if not candidates:
				return []
			return [min(candidates, key=lambda c: c.victoryFacet.victoryPoints if c.victoryFacet else 0)]
		if context == "bandit_trash":
			# Forced choice: sacrifice the less valuable Treasure.
			if not candidates:
				return []
			return [min(candidates, key=lambda c: c.treasureFacet.coinValue if c.treasureFacet else 0)]
		
		return []

	def ai_choose_pile(self, candidates, prompt, context):
		return max(candidates, key=lambda p: p.cardType.cost) if candidates else None

	def ai_choose_action(self, candidates, prompt, context):
		return candidates[0] if candidates else None

	def ai_choose_buy(self, candidates, prompt, context):
		import random
		if not candidates:
			return None
		action_affordable = [pile for pile in candidates if CardTypeKind.Action in pile.cardType.types]
		pool = action_affordable if action_affordable else candidates
		top_cost = max(pile.cardType.cost for pile in pool)
		tier = [pile for pile in pool if pile.cardType.cost == top_cost]
		return random.choice(tier)

	def play_action_card(self, card, game):
		if self.actions <= 0:
			return False
		if card not in self.hand.cards:
			return False
		self.actions -= 1
		self.hand.cards.remove(card)
		self.playArea.cards.append(card)
		self._resolve_action_effect(card, game)
		return True

	def _reveal_top(self, n):
		"""Reveal up to n cards off the top of this player's own deck
		(reshuffling discard in as needed), without adding them to hand.
		Used by Vassal/Sentry/Bandit-style effects."""
		import random
		revealed = []
		for _ in range(n):
			if not self.deck.cards:
				if not self.discardPile.cards:
					break
				self.deck.cards = self.discardPile.cards
				random.shuffle(self.deck.cards)
				self.discardPile.cards = []
				self.known_top_cards = []
			if self.deck.cards:
				card = self.deck.cards.pop()
				if self.known_top_cards and self.known_top_cards[0] is card:
					self.known_top_cards.pop(0)
				revealed.append(card)
		return revealed

	def _offer_moat_block(self, other, game):
		"""Give the attacked player (human or AI) a real chance to respond
		to an attack, instead of Moat silently auto-blocking it. Returns
		True if the attack was blocked."""
		moats = [c for c in other.hand.cards if c.name == "Moat"]
		if not moats:
			return False
		revealed = other.choose_cards_fn(moats, "An attack is being played against you -- reveal Moat to block it?",
		                                  "reveal_moat", -1)
		if not revealed:
			return False
		idx = game.players.index(other)
		print(f"      Player {idx} reveals Moat to block the attack!")
		return True

	def _resolve_action_effect(self, card, game):
		"""Just the effect logic, no action-cost/hand-removal bookkeeping,
		so Throne Room / Vassal can invoke a card's effect re-entrantly
		without re-paying its action cost or moving it again."""
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
				if self._offer_moat_block(other, game):
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
		elif name == "Poacher":
			self.actions += 1
			self.coins += 1
			self.draw(1)
			empty_piles = sum(1 for pile in game.supply.piles if pile.count <= 0)
			if empty_piles > 0 and self.hand.cards:
				n = min(empty_piles, len(self.hand.cards))
				to_discard = self.choose_cards_fn(list(self.hand.cards),
				                                   f"Poacher: discard {n} card(s) (one per empty supply pile)",
				                                   "poacher_discard", n)
				for c in to_discard:
					if c in self.hand.cards:
						self.hand.cards.remove(c)
						self.discardPile.cards.append(c)
		elif name == "Festival":
			self.actions += 2
			self.buys += 1
			self.coins += 2
		elif name == "Laboratory":
			self.actions += 1
			self.draw(2)
		elif name == "Council Room":
			self.buys += 1
			self.draw(4)
			for other in game.players:
				if other is not self:
					other.draw(1)
		elif name == "Moneylender":
			coppers = [c for c in self.hand.cards if c.name == "Copper"]
			if coppers:
				chosen = self.choose_cards_fn(coppers, "Moneylender: trash a Copper for +3 Coins?",
				                               "moneylender_trash", -1)
				if chosen:
					to_trash = chosen[0]
					self.hand.cards.remove(to_trash)
					game.trash.cards.append(to_trash)
					self.coins += 3
		elif name == "Chapel":
			if self.hand.cards:
				chosen = self.choose_cards_fn(list(self.hand.cards), "Chapel: trash up to 4 cards",
				                               "chapel_trash", -1)
				for c in chosen[:4]:
					if c in self.hand.cards:
						self.hand.cards.remove(c)
						game.trash.cards.append(c)
		elif name == "Artisan":
			gain_candidates = game.supply.eligible_piles(5, None)
			gain_pile = self.choose_pile_fn(gain_candidates, "Artisan: choose a card to gain (to hand)", "artisan_gain")
			if gain_pile is not None:
				gained = game.supply.take_from_pile(gain_pile)
				self.hand.cards.append(gained)
			if self.hand.cards:
				chosen = self.choose_cards_fn(list(self.hand.cards),
				                               "Artisan: put a card from hand onto your deck", "artisan_topdeck", 1)
				if chosen:
					to_topdeck = chosen[0]
					self.hand.cards.remove(to_topdeck)
					self.deck.cards.append(to_topdeck)
		elif name == "Library":
			import random
			while len(self.hand.cards) < 7:
				if not self.deck.cards:
					if not self.discardPile.cards:
						break
					self.deck.cards = self.discardPile.cards
					random.shuffle(self.deck.cards)
					self.discardPile.cards = []
				drawn = self.deck.cards.pop()
				if CardTypeKind.Action in drawn.types:
					skip = self.choose_cards_fn([drawn], "Library: skip this drawn Action card?", "library_skip", -1)
					if skip and drawn in skip:
						self.discardPile.cards.append(drawn)
						continue
				self.hand.cards.append(drawn)
		elif name == "Harbinger":
			self.actions += 1
			self.draw(1)
			if self.discardPile.cards:
				chosen = self.choose_cards_fn(list(self.discardPile.cards),
				                               "Harbinger: put a card from discard onto your deck?",
				                               "harbinger_topdeck", -1)
				if chosen:
					to_topdeck = chosen[0]
					if to_topdeck in self.discardPile.cards:
						self.discardPile.cards.remove(to_topdeck)
						self.deck.cards.append(to_topdeck)
		elif name == "Vassal":
			self.coins += 2
			revealed = self._reveal_top(1)
			if revealed:
				top = revealed[0]
				idx = game.players.index(self)
				print(f"      Player {idx} reveals {top.name} off the Vassal.")
				if CardTypeKind.Action in top.types:
					chosen = self.choose_cards_fn([top], "Vassal: play the revealed Action card?", "vassal_play", -1)
					if chosen and top in chosen:
						self.playArea.cards.append(top)
						print(f"      Player {idx} plays {top.name} (via Vassal): {CARD_DESCRIPTIONS.get(top.name, '')}")
						self._resolve_action_effect(top, game)
					else:
						self.discardPile.cards.append(top)
						print(f"      Player {idx} declines to play {top.name}; it's discarded.")
				else:
					self.discardPile.cards.append(top)
		elif name == "Sentry":
			self.actions += 1
			self.draw(1)
			revealed = self._reveal_top(2)
			if revealed:
				to_trash = self.choose_cards_fn(list(revealed), "Sentry: trash any of the revealed cards",
				                                 "sentry_trash", -1)
				for c in to_trash:
					if c in revealed:
						revealed.remove(c)
						game.trash.cards.append(c)
			if revealed:
				to_discard = self.choose_cards_fn(list(revealed),
				                                   "Sentry: discard any of the remaining revealed cards",
				                                   "sentry_discard", -1)
				for c in to_discard:
					if c in revealed:
						revealed.remove(c)
						self.discardPile.cards.append(c)
			if revealed:
				order = self.choose_cards_fn(list(revealed),
				                              "Sentry: choose card(s) to keep on top (rest go beneath, in order)",
				                              "sentry_reorder", -1)
				chosen_ids = {id(c) for c in order} if order else set()
				top_group = [c for c in order if id(c) in chosen_ids] if order else []
				rest_group = [c for c in revealed if id(c) not in chosen_ids]
				for c in rest_group:
					self.deck.cards.append(c)
				for c in top_group:
					self.deck.cards.append(c)
				# rest_group+top_group were just revealed to everyone and are
				# now back on top, soonest-drawn (top_group's last entry) first.
				self.known_top_cards = list(reversed(rest_group + top_group))
		elif name == "Witch":
			self.draw(2)
			for other in game.players:
				if other is self:
					continue
				if self._offer_moat_block(other, game):
					continue
				curse_piles = [pl for pl in game.supply.piles if pl.cardType.name == "Curse" and pl.count > 0]
				if curse_piles:
					gained = game.supply.take_from_pile(curse_piles[0])
					other.discardPile.cards.append(gained)
		elif name == "Bureaucrat":
			silver_piles = [pl for pl in game.supply.piles if pl.cardType.name == "Silver" and pl.count > 0]
			if silver_piles:
				gained = game.supply.take_from_pile(silver_piles[0])
				self.deck.cards.append(gained)
			for other in game.players:
				if other is self:
					continue
				if self._offer_moat_block(other, game):
					continue
				victory_cards = [c for c in other.hand.cards if CardTypeKind.Victory in c.types]
				if victory_cards:
					chosen = other.choose_cards_fn(victory_cards,
					                                "Bureaucrat: choose a Victory card to put on top of your deck",
					                                "bureaucrat_topdeck", 1)
					if chosen:
						to_topdeck = chosen[0]
						if to_topdeck in other.hand.cards:
							other.hand.cards.remove(to_topdeck)
							other.deck.cards.append(to_topdeck)
		elif name == "Bandit":
			gold_piles = [pl for pl in game.supply.piles if pl.cardType.name == "Gold" and pl.count > 0]
			if gold_piles:
				gained = game.supply.take_from_pile(gold_piles[0])
				self.discardPile.cards.append(gained)
			for other in game.players:
				if other is self:
					continue
				if self._offer_moat_block(other, game):
					continue
				revealed = other._reveal_top(2)
				eligible = [c for c in revealed if CardTypeKind.Treasure in c.types and c.name != "Copper"]
				to_trash = None
				if len(eligible) == 1:
					to_trash = eligible[0]
				elif len(eligible) > 1:
					chosen = other.choose_cards_fn(eligible, "Bandit: choose a Treasure to trash", "bandit_trash", 1)
					if chosen:
						to_trash = chosen[0]
				for c in revealed:
					if c is to_trash:
						game.trash.cards.append(c)
					else:
						other.discardPile.cards.append(c)
		elif name == "Throne Room":
			action_cards_in_hand = [c for c in self.hand.cards if CardTypeKind.Action in c.types]
			if action_cards_in_hand:
				target = self.choose_action_fn(action_cards_in_hand,
				                                "Throne Room: choose an action card to play twice",
				                                "throne_room_target")
				if target is not None and target in self.hand.cards:
					self.hand.cards.remove(target)
					self.playArea.cards.append(target)
					self._resolve_action_effect(target, game)
					self._resolve_action_effect(target, game)

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
		from ..rl.card_constants import COMPUTED_VALUE_VICTORY_CARDS
		all_cards = self.deck.cards + self.hand.cards + self.discardPile.cards + self.playArea.cards
		owned_card_count = len(all_cards)
		total = 0
		for c in all_cards:
			if CardTypeKind.Victory in c.types or CardTypeKind.Curse in c.types:
				computed = COMPUTED_VALUE_VICTORY_CARDS.get(c.name)
				if computed is not None:
					total += computed(owned_card_count)
				else:
					total += c.victoryFacet.victoryPoints
		return total

