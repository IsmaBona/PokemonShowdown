import asyncio
from poke_env.player.random_player import RandomPlayer
from poke_env import LocalhostServerConfiguration

async def main():
    # Crea un RandomPlayer conectado al servidor local
    random_player = RandomPlayer(
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1  # Solo una batalla a la vez
    )

    print("RandomPlayer conectado al servidor local.")
    print(f"Nombre de usuario: {random_player.username}")

    await random_player.send_challenges('Guest 1', 1)

# Ejecutar el evento principal
asyncio.run(main())