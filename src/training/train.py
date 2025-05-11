from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

from game.blackjack import Blackjack
from game.hand import Hand
from game.deck import Deck
from agents.q_learning_agent import QLearningAgent
from training.replayBuffer import PrioritizedReplayBuffer, Transition

@dataclass
class TrainingMetrics:
    episode_reward: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    splits: int = 0
    blackjacks: int = 0

class BlackjackTrainer:
    def __init__(self, agent: QLearningAgent) -> None:
        self.agent = agent
        self.batch_size = agent.batch_size
        
        # Performance tracking
        self.episode_rewards = []
        self.training_iterations = 0
        self.best_average_reward = float('-inf')
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(agent.buffer_size)
    
    def calculate_reward(self, player_value: int, dealer_value: int, 
                        playerHand: Hand, was_split: bool = False) -> float:
        base_reward = playerHand.bet
        
        # Handle player bust
        if player_value > 21:
            return -base_reward * 1.2 
        # Handle dealer bust
        if dealer_value > 21:
            return base_reward * 1.1 
        
        # Handle natural blackjack
        if player_value == 21 and len(playerHand.getHand()) == 2 and not was_split:
            return base_reward * 1.5
        
        # Handle regular win/loss
        if player_value > dealer_value:
            return base_reward * (1.1 if player_value >= 20 else 1.0)
        elif player_value < dealer_value:
            return -base_reward
        else:
            return 0

    def train_episode(self) -> TrainingMetrics:
        game = Blackjack()
        metrics = TrainingMetrics()
        dealerHand = Hand()
        playerHand = Hand()
        gameDeck = Deck()
        
        game.deal(gameDeck, dealerHand, playerHand)
        hands_to_play = [(playerHand, 0)]  # (hand, split_depth)
        episode_transitions = []
        
        while hands_to_play:
            current_hand, split_depth = hands_to_play.pop(0)
            done = False
            
            while not done:
                state = self.agent.getState(current_hand, dealerHand, gameDeck)
                action = self.agent.getAction(current_hand, dealerHand, gameDeck, split_depth)
                
                if action == 2:  # Split
                    metrics.splits += 1
                    new_hand = current_hand.splitHand()
                    
                    try:
                        current_hand.addCard(gameDeck.draw())
                        new_hand.addCard(gameDeck.draw())
                    except ValueError:
                        print("Warning: Deck ran out of cards during split")
                        break
                        
                    hands_to_play.append((new_hand, split_depth + 1))
                    
                    next_state = self.agent.getState(current_hand, dealerHand, gameDeck)
                    transition = Transition(state, action, 0, next_state, False)
                    episode_transitions.append(transition)
                    done = True
                    
                elif action == 1:  # Hit
                    try:
                        current_hand.addCard(gameDeck.draw())
                    except ValueError:
                        print("Warning: Deck ran out of cards during hit")
                        break
                        
                    next_state = self.agent.getState(current_hand, dealerHand, gameDeck)
                    
                    if next_state[0] > 21:  # Bust
                        reward = -1 * current_hand.bet
                        metrics.losses += 1
                        transition = Transition(state, action, reward, next_state, True)
                        episode_transitions.append(transition)
                        metrics.episode_reward += reward
                        done = True
                    else:
                        transition = Transition(state, action, 0, next_state, False)
                        episode_transitions.append(transition)
                
                else:  # Stand
                    if not hands_to_play:
                        dealer_value = game.getHandValue(dealerHand)
                        while dealer_value < 17:
                            try:
                                dealerHand.addCard(gameDeck.draw())
                            except ValueError:
                                print("Warning: Deck ran out of cards during dealer play")
                                break
                            dealer_value = game.getHandValue(dealerHand)
                    
                    player_value = game.getHandValue(current_hand)
                    dealer_value = game.getHandValue(dealerHand)
                    
                    reward = self.calculate_reward(player_value, dealer_value, current_hand)
                    metrics.episode_reward += reward
                    
                    if reward > 0:
                        metrics.wins += 1
                    elif reward < 0:
                        metrics.losses += 1
                    else:
                        metrics.draws += 1
                        
                    if player_value == 21 and len(current_hand.getHand()) == 2:
                        metrics.blackjacks += 1
                    
                    next_state = self.agent.getState(current_hand, dealerHand, gameDeck)
                    transition = Transition(state, action, reward, next_state, True)
                    episode_transitions.append(transition)
                    done = True
        
        #process transitions and update replay buffer
        for transition in episode_transitions:
            error = self.agent.update_Q(transition.state, transition.action,
                                     transition.reward, transition.next_state,
                                     transition.terminal)
            self.replay_buffer.add(transition, error)
        
        return metrics

    def train(self, numEpisodes: int = 10000) -> List[TrainingMetrics]:
        all_metrics = []
        
        for _ in range(numEpisodes):
            #train single episode
            metrics = self.train_episode()
            all_metrics.append(metrics)
            
            # Experience replay
            if len(self.replay_buffer) >= self.batch_size:
                samples, indices, weights = self.replay_buffer.sample(self.batch_size)
                errors = []
                
                for transition, weight in zip(samples, weights):
                    error = self.agent.update_Q(transition.state, transition.action,
                                             transition.reward, transition.next_state,
                                             transition.terminal)
                    errors.append(error * weight)
                
                self.replay_buffer.update_priorities(indices, errors)
            
            # Update training state
            self.episode_rewards.append(metrics.episode_reward)
            self.agent.decay_epsilon()
            self.training_iterations += 1
            
            #update best average reward
            if len(self.episode_rewards) >= 100:
                recent_avg = np.mean(self.episode_rewards[-100:])
                self.best_average_reward = max(self.best_average_reward, recent_avg)
        
        return all_metrics