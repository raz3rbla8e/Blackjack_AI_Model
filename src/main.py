import sys
from training.optimizer import HyperparameterOptimizer
from agents.q_learning_agent import QLearningAgent
from training.train import BlackjackTrainer
from game.blackjack import Blackjack
from game.hand import Hand
from game.deck import Deck
from agents.evaluation import evaluate_agent
from utils.config import TrainingConfig

def play_with_saved_model(model_path: str, num_games: int = 100) -> None:
    #load json for saved agent
    agent = QLearningAgent.load(model_path)
    metrics = evaluate_agent(agent, num_games)
    #evaluate the saved agent across num_games
    print("\nFinal Statistics:")
    print(f"Games played: {num_games}")
    print(f"Win rate: {metrics['win_rate']:.1f}%")
    print(f"Average reward per game: {metrics['avg_reward']:.3f}")
    print(f"Wins: {metrics['wins']} ({(metrics['wins']/num_games)*100:.1f}%)")
    print(f"Losses: {metrics['losses']} ({(metrics['losses']/num_games)*100:.1f}%)")
    print(f"Draws: {metrics['draws']} ({(metrics['draws']/num_games)*100:.1f}%)")
    print(f"Blackjacks: {metrics['blackjacks']} ({(metrics['blackjacks']/num_games)*100:.1f}%)")
    print(f"Busts: {metrics['busts']} ({(metrics['busts']/num_games)*100:.1f}%)")
    print(f"Splits: {metrics['splits']} ({(metrics['splits']/num_games)*100:.1f}%)")

def main():
    #training mode
    if len(sys.argv) == 1:
        print("Starting training mode...")
        #init config
        config = TrainingConfig()
        #optimize
        optimizer = HyperparameterOptimizer(config)
        optimizer.run_optimization()
    #saved model mode
    elif len(sys.argv) == 2:
        model_path = sys.argv[1]
        print(f"Loading model from: {model_path}")
        play_with_saved_model(model_path, num_games=100000)

if __name__ == "__main__":
    main()