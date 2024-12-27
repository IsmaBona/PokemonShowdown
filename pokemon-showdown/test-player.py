import asyncio
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


# Load the type chart
def load_type_chart(file_path: str) -> Dict[str, Dict[str, float]]:
    with open(file_path, "r") as file:
        type_chart = json.load(file)
    return type_chart


class SimpleRLPlayer(Player):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, **kwargs):
        super().__init__(**kwargs)
        self.type_chart = load_type_chart("gen8typechart.json")
        self.np_random = None

        # Q-learning parameters
        self.q_table = {}  # Q-table for state-action pairs
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay for epsilon after each battle
        self.action_space_size = 4  # Maximum moves to consider (adjust if needed)

        # State and action trackers
        self.last_state = None
        self.last_action = None

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def parse_message(self, message):
        try:
            super().parse_message(message)
        except NotImplementedError as e:
            print(f"Mensaje ignorado: {message}")

    def embed_battle(self, battle: AbstractBattle) -> Tuple:
        moves_base_power = [-1] * 4
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power // 10 if move.base_power else -1

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted])
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted])

        return tuple(moves_base_power + [fainted_mon_team, fainted_mon_opponent])

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0]
        high = [3, 3, 3, 3, 6, 6]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def choose_move(self, battle: AbstractBattle) -> Move:
        """
        Select the next move based on Q-learning policy (epsilon-greedy).
        """
        state = self.embed_battle(battle)

        # Initialize Q-values for the state if not already present
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            action = random.randint(0, self.action_space_size - 1)
        else:
            # Exploitation: choose the action with the highest Q-value
            action = np.argmax(self.q_table[state])

        # Translate action to a move
        if battle.available_moves and action < len(battle.available_moves):
            selected_move = battle.available_moves[action]
            self.last_action = action
            self.last_state = state
            return self.create_order(selected_move)
        elif battle.available_switches:
            return self.choose_random_move(battle)
        else:
            return self.choose_random_move(battle)

    def update_q_table(self, reward: float, current_state: Tuple):
        """
        Update Q-values using the Q-learning formula.
        """
        if self.last_state is None or self.last_action is None:
            return  # Skip if there's no previous state-action pair

        # Ensure the current state is in the Q-table
        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(self.action_space_size)

        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[current_state])
        self.q_table[self.last_state][self.last_action] += self.alpha * (
            reward
            + self.gamma * self.q_table[current_state][best_next_action]
            - self.q_table[self.last_state][self.last_action]
        )

        # Reset last state and action
        self.last_state = None
        self.last_action = None

    def reward_battle(self, battle: AbstractBattle, reward: float):
        """
        Override to include Q-table updates at the end of a battle.
        """
        current_state = self.embed_battle(battle)
        self.update_q_table(reward, current_state)


# Define the asynchronous function
async def main():
    p1 = RandomPlayer(battle_format="gen8randombattle")
    p2 = SimpleRLPlayer(battle_format="gen8randombattle")

    # Simulate a battle
    await p1.battle_against(p2, n_battles=10)

    # Print Q-table after battles
    print("Q-table after training:")
    for state, values in p2.q_table.items():
        print(f"State: {state}, Q-values: {values}")


# Execute the asynchronous function
asyncio.run(main())