import os
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
            #selected_move = battle.available_moves[action]
            #return self.create_order(selected_move)
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


async def train_rl_player(
    n_training_battles=1000,
    checkpoint_interval=100,
    checkpoint_dir="checkpoints",
    log_interval=50,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize players
    rl_player = SimpleRLPlayer(battle_format="gen8randombattle")
    random_player = RandomPlayer(battle_format="gen8randombattle")

    for battle in range(1, n_training_battles + 1):
        await rl_player.battle_against(random_player, n_battles=1)

        # Decay epsilon
        rl_player.epsilon *= rl_player.epsilon_decay

        # Logging
        if battle % log_interval == 0:
            print(
                f"Battle {battle}/{n_training_battles} - Epsilon: {rl_player.epsilon:.4f} - "
                f"Q-table size: {len(rl_player.q_table)}"
            )

        # Save checkpoint
        if battle % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_battle_{battle}.json")

            # Convert tuple keys to strings for JSON serialization
            q_table_serializable = {str(k): v.tolist() for k, v in rl_player.q_table.items()}

            with open(checkpoint_path, "w") as f:
                json.dump(
                    {
                        "q_table": q_table_serializable,
                        "epsilon": rl_player.epsilon,
                    },
                    f,
                )
            print(f"Checkpoint saved at {checkpoint_path}")

    print("Training complete!")


# Run training
asyncio.run(train_rl_player())
