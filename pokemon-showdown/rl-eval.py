from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player
from poke_env.environment import AbstractBattle
import asyncio

import os
import random
import numpy as np
import json
from gymnasium.spaces import Box, Space
from poke_env.data.static import typechart
from typing import Dict, Tuple
from poke_env.environment.move import Move
from poke_env.player.random_player import RandomPlayer
from poke_env.environment import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from poke_env.player import MaxBasePowerPlayer


# Load the type chart
def load_type_chart(file_path: str) -> Dict[str, Dict[str, float]]:
    with open(file_path, "r") as file:
        type_chart = json.load(file)
    return type_chart


class SimpleRLPlayer(Player):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, **kwargs):
        super().__init__(**kwargs)
        self.type_chart = load_type_chart("gen8typechart.json")
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.action_space_size = 4  # Adjust if needed

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> tuple:
        moves_base_power = [-1] * 4
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power // 10 if move.base_power else -1

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted])
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted])

        return tuple(moves_base_power + [fainted_mon_team, fainted_mon_opponent])

    def choose_move(self, battle: AbstractBattle) -> Move:
        state = self.embed_battle(battle)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)

        if random.random() < self.epsilon:
            action = random.randint(0, self.action_space_size - 1)
        else:
            action = np.argmax(self.q_table[state])

        if battle.available_moves and action < len(battle.available_moves):
            selected_move = battle.available_moves[action]
            return self.create_order(selected_move)
        elif battle.available_switches:
            return self.choose_random_move(battle)
        else:
            return self.choose_random_move(battle)

    def update_q_table(self, last_state, last_action, reward, current_state):
        if last_state not in self.q_table:
            self.q_table[last_state] = np.zeros(self.action_space_size)
        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(self.action_space_size)

        best_next_action = np.argmax(self.q_table[current_state])
        self.q_table[last_state][last_action] += self.alpha * (
            reward + self.gamma * self.q_table[current_state][best_next_action] - self.q_table[last_state][last_action]
        )
    
def load_pretrained_rl_player(checkpoint_path: str, **kwargs) -> SimpleRLPlayer:
    """
    Load a pre-trained SimpleRLPlayer from a checkpoint.
    """
    pre_trained_player = SimpleRLPlayer(**kwargs)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        
        # Deserialize Q-table
        pre_trained_player.q_table = {
            eval(k): np.array(v) for k, v in checkpoint["q_table"].items()
        }
        pre_trained_player.epsilon = checkpoint["epsilon"]
        print(f"Loaded pre-trained player from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return pre_trained_player

# Load checkpoint function (similar to your training code)
def load_checkpoint(checkpoint_path):
    try:
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
            q_table = {
                tuple(map(int, k[1:-1].split(", "))): np.array(v)
                for k, v in checkpoint_data["q_table"].items()
            }
            epsilon = checkpoint_data["epsilon"]
            return q_table, epsilon
    except FileNotFoundError:
        print(f"Checkpoint file {checkpoint_path} not found.")
        return None, None

# Evaluation function
async def evaluate_rl_agent(
    n_eval_battles=100,
    checkpoint_path="checkpoints/checkpoint_latest.json",
    opponent_checkpoint=None,
):
    # Load the trained RL agent from checkpoint
    print("Loading the trained RL agent from checkpoint...")
    q_table, epsilon = load_checkpoint(checkpoint_path)

    # Initialize the RL agent
    rl_player = SimpleRLPlayer(battle_format="gen8randombattle")
    if q_table is not None:
        rl_player.q_table = q_table
        rl_player.epsilon = epsilon
        print(f"Loaded training checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found. Evaluation cannot proceed.")
        return

    # Initialize the BaseMaxPower opponent
    base_max_power_player = MaxBasePowerPlayer(battle_format="gen8randombattle")

    # Win tracking
    rl_wins = 0
    opponent_wins = 0

    for battle in range(1, n_eval_battles + 1):
        if battle:
            await rl_player.battle_against(base_max_power_player, n_battles=1)

            # Update win counters
            rl_wins = rl_player.n_won_battles
            opponent_wins = base_max_power_player.n_won_battles
            print(f"Evaluating battle {battle}/{n_eval_battles}...")
            print(f"RL Agent Wins: {rl_wins} | BaseMaxPower Opponent Wins: {opponent_wins} ")
        else:
            print(f"Battle {battle}/{n_eval_battles} did not return a valid result. Skipping...")

    # Final results after all evaluation battles
    print("\nEvaluation complete!")
    print(
        f"Total battles: {n_eval_battles} | RL Agent Wins: {rl_wins} | BaseMaxPower Opponent Wins: {opponent_wins}"
    )
    print(f"Final win rate: {(rl_wins / n_eval_battles) * 100:.2f}%")

# Run the evaluation
asyncio.run(evaluate_rl_agent())
