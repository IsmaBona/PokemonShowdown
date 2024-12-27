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


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        
        # Deserialize Q-table
        q_table = {eval(k): np.array(v) for k, v in checkpoint["q_table"].items()}
        epsilon = checkpoint["epsilon"]
        return q_table, epsilon
    else:
        return None, None
    
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


async def train_rl_player(
    n_training_battles=1000,
    checkpoint_interval=100,
    checkpoint_dir="checkpoints",
    log_interval=50,
    opponent_checkpoint=None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.json")

    # Check for an existing checkpoint for the training player
    q_table, epsilon = load_checkpoint(checkpoint_path)

    # Initialize players
    rl_player = SimpleRLPlayer(battle_format="gen8randombattle")
    if opponent_checkpoint:
        # Use a pre-trained RL player as the opponent
        opponent_player = load_pretrained_rl_player(
            opponent_checkpoint, battle_format="gen8randombattle"
        )
    else:
        # Use a RandomPlayer as the opponent
        opponent_player = RandomPlayer(battle_format="gen8randombattle")

    if q_table is not None:
        rl_player.q_table = q_table
        rl_player.epsilon = epsilon
        print(f"Loaded training checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found for training player. Starting from scratch.")

    # Win tracking
    rl_wins = 0
    opponent_wins = 0

    for battle in range(1, n_training_battles + 1):
        await rl_player.battle_against(opponent_player, n_battles=1)

        # Update win counters
        rl_wins = rl_player.n_won_battles
        opponent_wins = opponent_player.n_won_battles

        # Decay epsilon
        rl_player.epsilon *= rl_player.epsilon_decay

        # Logging
        if battle % log_interval == 0:
            print(
                f"Battle {battle}/{n_training_battles} - Epsilon: {rl_player.epsilon:.4f} - "
                f"Q-table size: {len(rl_player.q_table)}"
            )

        # Save checkpoint and print win stats
        if battle % checkpoint_interval == 0 or battle == n_training_battles:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.json")

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
            print(
                f"After {battle} battles: RL Agent wins: {rl_wins}, Opponent wins: {opponent_wins}"
            )

    print("Training complete!")

# Run training
opponent_checkpoint_path = "checkpoints/pretrained_opponent.json"  # Path to pre-trained opponent checkpoint

asyncio.run(train_rl_player(opponent_checkpoint=opponent_checkpoint_path))