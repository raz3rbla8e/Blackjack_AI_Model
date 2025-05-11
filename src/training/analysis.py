import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from agents.base_agent import BaseAgent

#creates strategy visualizations for report
def analyze_agent_strategy(agent: BaseAgent, results_dir: str) -> None:
    #get strategy matrices directly from agent
    hard_totals, soft_totals, pairs = agent.get_strategy_matrices()
    
    # Plot strategy matrices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    #plot hard totals
    sns.heatmap(hard_totals, ax=ax1, cmap='RdYlBu',
               xticklabels=range(2, 12),
               yticklabels=range(12, 22))
    ax1.set_title('Hard Totals Strategy')
    ax1.set_xlabel('Dealer Upcard')
    ax1.set_ylabel('Player Total')
    
    #plot soft totals
    sns.heatmap(soft_totals, ax=ax2, cmap='RdYlBu',
               xticklabels=range(2, 12),
               yticklabels=range(12, 22))
    ax2.set_title('Soft Totals Strategy')
    ax2.set_xlabel('Dealer Upcard')
    ax2.set_ylabel('Player Total')
    
    #plot pairs
    sns.heatmap(pairs, ax=ax3, cmap='RdYlBu',
               xticklabels=range(2, 12),
               yticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'])
    ax3.set_title('Pairs Strategy')
    ax3.set_xlabel('Dealer Upcard')
    ax3.set_ylabel('Pair Value')
    #save the figure
    plt.tight_layout()
    plt.savefig(f'{results_dir}/agent_strategy.png', 
               bbox_inches='tight', dpi=300)
    plt.close()