from typing import List
from game.deck import Deck
from game.hand import Hand
from game.card import Card
from agents.base_agent import BaseAgent

class Blackjack:
    def __init__(self):
        #this is kind of arbitrary, but from what i found casinos let you split a max number of times.
        #dont want the agent to learn to infinitely split hands
        self.MAX_SPLITS = 3  
    
    #get the value of a hand
    def getHandValue(self, hand: Hand) -> int:
        cards = hand.getHand()
        value = 0
        aces = 0
        
        for card in cards:
            cardValue = card.getValue()
            if cardValue == 1:
                aces += 1
                value += 11
            else:
                value += cardValue
                
        #aces can have different values if they would cause a bust
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
            
        return value

    def deal(self, gameDeck: Deck, dealerHand: Hand, playerHand: Hand) -> None:
        #kind of arbitrary here, but this is how cards are dealt in blackjack
        playerHand.addCard(gameDeck.draw())
        dealerHand.addCard(gameDeck.draw())
        playerHand.addCard(gameDeck.draw())
        dealerHand.addCard(gameDeck.draw())
    
    #play a single hand
    #returns total reward
    def play_hand(self, gameDeck: Deck, playerHand: Hand, dealerHand: Hand, 
                 agent: BaseAgent, split_count: int = 0) -> float:
        while True:
            action = agent.getAction(playerHand, dealerHand, gameDeck)
            #split
            if action == 2 and split_count < self.MAX_SPLITS:  
                #ceate new hand from split
                newHand = playerHand.splitHand()
                
                #deal
                playerHand.addCard(gameDeck.draw())
                newHand.addCard(gameDeck.draw())
                
                #play both hands
                reward1 = self.play_hand(gameDeck, playerHand, dealerHand, 
                                      agent, split_count + 1)
                reward2 = self.play_hand(gameDeck, newHand, dealerHand, 
                                      agent, split_count + 1)
                
                return reward1 + reward2
            #hit
            elif action == 1:
                playerHand.addCard(gameDeck.draw())
                if self.getHandValue(playerHand) > 21:
                    return -playerHand.bet
            #stand
            else:  
                break
        
        #only play out dealer hand if this is the last split hand
        if split_count == 0:
            dealer_value = self.getHandValue(dealerHand)
            while dealer_value < 17:
                dealerHand.addCard(gameDeck.draw())
                dealer_value = self.getHandValue(dealerHand)
        
        player_value = self.getHandValue(playerHand)
        dealer_value = self.getHandValue(dealerHand)
        
        #handle player/dealer blackjack
        player_blackjack = player_value == 21 and len(playerHand.getHand()) == 2
        dealer_blackjack = dealer_value == 21 and len(dealerHand.getHand()) == 2
        
        #modify reward for player blackjack
        if player_blackjack and not dealer_blackjack:
            return 1.5 * playerHand.bet
        elif dealer_blackjack and not player_blackjack:
            return -playerHand.bet
        elif player_blackjack and dealer_blackjack:
            return 0
        
        #decision tree for normal final evaluation (no blackjack)
        if dealer_value > 21:
            return playerHand.bet
        elif player_value > dealer_value:
            return playerHand.bet
        elif player_value < dealer_value:
            return -playerHand.bet
        else:
            return 0