from abc import ABC, abstractmethod
from ..cards.card import Card

class CardZone(ABC):

	def __init__(self):
		self.cards: Card = []

