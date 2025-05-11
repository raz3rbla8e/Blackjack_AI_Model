from typing import List
from game.card import Card

class Hand:
    def __init__(self):
        self.cards = []
        self.bet = 1  # this isnt really implemented yet, idk if we want to.
        #if we do decide to implement educated betting, refer to kelly criterion
        self.canSplit = True  # Whether this hand can still be split

    def getHand(self) -> List[Card]:
        return self.cards
    
    def addCard(self, card: Card) -> None:
        self.cards.append(card)
    
    #chekc if the hand can be split
    def canBeSplit(self) -> bool:
        return (len(self.cards) == 2 and 
                self.cards[0].getValue() == self.cards[1].getValue() and 
                self.canSplit)
    
    #split a splittable hand
    def splitHand(self) -> 'Hand':
        if not self.canBeSplit():
            raise ValueError("CANT SPLIT")
        
        newHand = Hand()
        newHand.addCard(self.cards.pop())
        newHand.bet = self.bet
        newHand.canSplit = True
        return newHand