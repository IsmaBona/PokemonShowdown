import asyncio
from poke_env.player.random_player import RandomPlayer

# Define an asyncronous funtion
async def main():
	# Create 2 RandomPlayers
    p1 = RandomPlayer(battle_format="gen8randombattle")
    p2 = RandomPlayer(battle_format="gen8randombattle")

    # Simulate a battle between 2 RandomPlayers
    await p1.battle_against(p2, n_battles=1)

# Execute the asyncronous function
asyncio.run(main())
