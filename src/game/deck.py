from random import shuffle
from game.card import Card

#Represents a deck of cards
class Deck:
    def __init__(self):
        self.cards = []
        self.running_count = 0
        self.cards_remaining = 52
        
        suits = ["hearts", "diamonds", "clubs", "spades"]
        ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # 10 repeated for J,Q,K
        
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))
        shuffle(self.cards)

    def draw(self) -> Card:
        if not self.cards:
            raise ValueError("attempted to draw from empty deck")
            
        card = self.cards.pop()
        self.running_count += card.getCountValue()
        self.cards_remaining -= 1
        return card
    
    def getTrueCount(self) -> float:
        decks_remaining = max(self.cards_remaining / 52, 0.5)  # avoid division by zero
        return self.running_count / decks_remaining
    #call init to reset/"shuffle"
    def shuffle_deck(self) -> None:
        self.__init__() 
    
    #return # of cards left in deck
    def cards_left(self) -> int:
        return len(self.cards)