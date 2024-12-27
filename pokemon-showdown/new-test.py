import asyncio
import random

import numpy as np
import json
from gymnasium.spaces import Box, Space
from poke_env.data.static import typechart
from typing import Dict
from poke_env.environment.move import Move

from poke_env.player.random_player import RandomPlayer
from poke_env.environment import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

# Cargar el archivo JSON como un diccionario
def load_type_chart(file_path: str) -> Dict[str, Dict[str, float]]:
    with open(file_path, "r") as file:
        type_chart = json.load(file)  # Cargar el archivo JSON como diccionario
    return type_chart
    
class SimpleRLPlayer(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type_chart = load_type_chart("gen8typechart.json")
        self.np_random = None

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def parse_message(self, message):
        """Manejo de mensajes no reconocidos."""
        try:
            super().parse_message(message)
        except NotImplementedError as e:
            print(f"Mensaje ignorado: {message}")

    def embed_battle(self, battle: AbstractBattle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=self.type_chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    """
    def choose_move(self, battle: AbstractBattle):
        # Get the battle state embedding (i.e., observation)
        state = self.embed_battle(battle)
        print(state)
        
        # For now, let's implement a simple random move choice based on available moves.
        # In a real RL scenario, you would use the state here to select the best move
        # using a policy (for example, a Q-table or a neural network).

        if self.format_is_doubles:
            return self.choose_doubles_move(battle)  # type: ignore
        else:
            if battle.available_moves:
                print("Move available")
                print(battle.available_moves)
                best_move = max(battle.available_moves, key=lambda move: move.base_power)
                return self.create_order(best_move)
            print("Move not available")
            print(battle.available_moves)
            return self.choose_random_move(battle)
    """
    def choose_move(self, battle: AbstractBattle) -> Move:
        """
        Select the next move based on the battle state embedding.

        This method uses the battle's state embedding to decide the best move.
        For demonstration purposes, this will use simple logic based on the embedding.
        """
        # Get the battle state embedding (i.e., observation)
        state = self.embed_battle(battle)
        
        # Example logic: prioritize moves with the highest damage multiplier
        # If there are no moves, fallback to other choices
        if battle.available_moves:
            # Create a list of tuples (move, predicted value based on embedding)
            move_values = []
            for i, move in enumerate(battle.available_moves):
                # Assume the state embedding's damage multiplier influences decision
                predicted_value = state[4 + i]  # Access the damage multiplier from embedding
                move_values.append((move, predicted_value))
            
            # Choose the move with the highest predicted value
            best_move = max(move_values, key=lambda x: x[1])[0]
            return self.create_order(best_move)
        
        # If no moves are available, try switching
        elif battle.available_switches:
            return self.choose_random_move(battle)
        
        # If neither moves nor switches are available, forfeit
        else:
            return self.choose_random_move(battle)


# Define an asyncronous funtion
async def main():
	# Create 2 RandomPlayers
    p1 = RandomPlayer(battle_format="gen8randombattle")
    p2 = SimpleRLPlayer(battle_format="gen8randombattle")

    # Simulate a battle between 2 RandomPlayers
    await p1.battle_against(p2, n_battles=1)

# Execute the asyncronous function
asyncio.run(main())


