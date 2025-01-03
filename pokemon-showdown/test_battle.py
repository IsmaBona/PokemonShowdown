import asyncio
from poke_env.player.random_player import RandomPlayer
from poke_env.player import Player
from eval_player import PlayerPerformance

# Define an asyncronous funtion
async def main():
	# Create 2 RandomPlayers
    p1 = Player(username="RandomPlayer 1", battle_format="gen8randombattle")
    p2 = RandomPlayer(battle_format="gen8randombattle")

    # Simulate a battle between 2 RandomPlayers
    await p1.battle_against(p2, n_battles=1)
    player = PlayerPerformance(p1)
    player.analyze_battle_history()
    player.evaluate_performance()


# Execute the asyncronous function

asyncio.run(main())
