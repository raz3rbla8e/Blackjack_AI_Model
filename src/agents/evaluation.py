from typing import Dict, Any
import numpy as np
from game.blackjack import Blackjack
from game.hand import Hand
from game.deck import Deck
from agents.base_agent import BaseAgent

def evaluate_agent(agent: BaseAgent, num_games: int = 100) -> dict:
    #evaluate agent over num_games
    game = Blackjack()
    metrics = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'splits': 0,
        'blackjacks': 0,
        'busts': 0,
        'total_reward': 0,
        'hand_outcomes': []
    }
    
    for _ in range(num_games):
        dealerHand = Hand()
        playerHand = Hand()
        gameDeck = Deck()
        game.deal(gameDeck, dealerHand, playerHand)
        state = agent.getState(playerHand, dealerHand, gameDeck)
        initial_action = agent.getAction(playerHand, dealerHand, gameDeck)

        if initial_action == 2:
            metrics['splits'] += 1
        
        reward = game.play_hand(gameDeck, playerHand, dealerHand, agent)
        metrics['total_reward'] += reward
        metrics['hand_outcomes'].append(reward)
        
        player_value = agent.getHandValue(playerHand.getHand())
        dealer_value = agent.getHandValue(dealerHand.getHand())
        
        if player_value > 21:
            metrics['busts'] += 1
        elif player_value == 21 and len(playerHand.getHand()) == 2:
            metrics['blackjacks'] += 1
        
        if reward > 0:
            metrics['wins'] += 1
        elif reward < 0:
            metrics['losses'] += 1
        else:
            metrics['draws'] += 1
    
    #calculate averages
    metrics['win_rate'] = (metrics['wins'] + 0.5 * metrics['draws']) / num_games * 100
    metrics['avg_reward'] = metrics['total_reward'] / num_games
    metrics['split_rate'] = (metrics['splits'] / num_games) * 100
    metrics['blackjack_rate'] = (metrics['blackjacks'] / num_games) * 100
    metrics['bust_rate'] = (metrics['busts'] / num_games) * 100
    metrics['reward_variance'] = np.var(metrics['hand_outcomes'])
    
    return metrics