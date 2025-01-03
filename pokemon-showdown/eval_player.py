from poke_env.player.player import Player
from poke_env.environment import AbstractBattle
import asyncio
from typing import Dict

class PlayerPerformance(Player):
    def __init__(self, username, battle_format, *args, **kwargs):
        # Asegúrate de pasar solo los parámetros esperados por Player
        super().__init__(battle_format, *args, **kwargs)
        self.games_played = 0
        self.games_won = 0
        self.damage_given = 0
        self.damage_taken = 0
        self.total_moves = 0
        self.name = username

        

    def choose_move(self, battle: AbstractBattle):
        """
        Este método es necesario para cumplir con la interfaz de la clase Player.
        Puedes implementar un jugador aleatorio o una lógica básica para elegir un movimiento.
        """
        # Este ejemplo simplemente elige un movimiento aleatorio de los movimientos disponibles
        return battle.current_pokemon.choose_move(battle.available_moves)
        
    def analyze_battle_history(self):
        """
        Procesa el historial de batallas existente y actualiza las estadísticas acumulativas.
        """
        
        if not self.battles:
            print("No hay batallas en el historial para analizar.")
        else:
            for battle_id, battle in self.battles.items():
                self.games_played = self.n_finished_battles()
                
                # Determinar si el jugador ganó la batalla
                self.games_won = self.n_won_battles()

                # Calcular daño infligido
                total_battle_damage = sum(pokemon._current_hp for pokemon in battle._opponent_team.values())
                self.damage_given += total_battle_damage

                # Calcular daño recibido
                total_damage_taken = sum(pokemon._current_hp for pokemon in battle._team.values())
                self.damage_taken += total_damage_taken

                # Calcular movimientos realizados
                total_battle_moves = sum(len(pokemon._moves) for pokemon in battle.team.values())
                self.total_moves += total_battle_moves
            if battle.player_username:
                self.name = battle.player_username
            else:
                self.name = "None"

    def on_battle_end(self, battle: AbstractBattle):
        """
        Método llamado cuando una batalla termina. Actualiza estadísticas dinámicamente.
        """
        self.games_played = self.n_finished_battles
        # Determinar si el jugador ganó la batalla
        self.games_won = self.n_won_battles

        # Calcular daño infligido
        total_battle_damage = sum(pokemon._current_hp for pokemon in battle.opponent_team.values())
        self.damage_given += total_battle_damage

        # Calcular daño recibido
        total_damage_taken = sum(pokemon._current_hp for pokemon in battle.team.values())
        self.damage_taken += total_damage_taken

        # Calcular movimientos realizados
        total_battle_moves = sum(len(pokemon._moves) for pokemon in battle.team.values())
        self.total_moves += total_battle_moves

    # Métodos de evaluación
    def win_rate(self):
        return self.games_won / self.games_played if self.games_played else 0
    
    def average_damage(self):
        return self.damage_given / self.games_played if self.games_played else 0

    def delta_damage(self):
        return self.damage_given - self.damage_taken if self.games_played else 0

    def average_moves(self):
        return self.total_moves / self.games_played if self.games_played else 0

    def average_damage_move(self):
        return self.damage_given / self.total_moves if self.total_moves else 0

    def evaluate_performance(self):
        print(f"Player name: {self.name}")
        print(f"Win Rate: {self.win_rate():.2f}")
        print(f"Average Damage: {self.average_damage():.2f}")
        print(f"Delta Damage: {self.delta_damage():.2f}")
        print(f"Average Moves per Game: {self.average_moves():.2f}")
        print(f"Average Damage per Move: {self.average_damage_move():.2f}")

# Integración con test.py
def main():

    player = PlayerPerformance(name=Player, battle_format="gen8randombattle")
    
    # Analiza el historial existente
    player.analyze_battle_history()
    player.evaluate_performance()

if __name__ == "__main__":
    asyncio.run(main())