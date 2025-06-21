# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:25:38 2024

@author: Sebastián Serra
"""

from statsbombpy import sb
import pandas as pd

# Obtener competiciones y partidos
comps = sb.competitions()  # competition_id = 11, season_id = 90 LaLiga 20/21
matches = sb.matches(competition_id = 11, season_id = 90)

# Lista para almacenar los DataFrames agrupados de cada partido
grouped_events_list = []

# Columnas que queremos seleccionar y sus métodos de agregación
columns_agg_methods = {
    'position': 'first',  # Mantiene la primera posición encontrada
    'pass_recipient': 'count',
    'goalkeeper_outcome': 'count',
    '50_50': 'count',
    'bad_behaviour_card': 'count',
    'ball_receipt_outcome': 'count',
    'ball_recovery_offensive': 'count',
    'ball_recovery_recovery_failure': 'count',
    'clearance_aerial_won': 'count',
    'clearance_head': 'count',
    'clearance_left_foot': 'count',
    'clearance_right_foot': 'count',
    'counterpress': 'count',
    'dribble_nutmeg': 'count',
    'dribble_outcome': 'count',
    'dribble_overrun': 'count',
    'duel_outcome': 'count',
    'foul_committed_advantage': 'count',
    'foul_committed_card': 'count',
    'foul_committed_offensive': 'count',
    'foul_committed_type': 'count',
    'foul_won_advantage': 'count',
    'foul_won_defensive': 'count',
    'miscontrol_aerial_won': 'count',
    'out': 'count',
    'pass_aerial_won': 'count',
    'pass_assisted_shot_id': 'count',
    'pass_cross': 'count',
    'pass_cut_back': 'count',
    'pass_deflected': 'count',
    'pass_inswinging': 'count',
    'pass_length': 'mean',
    'pass_no_touch': 'count',
    'pass_outcome': 'count',
    'pass_outswinging': 'count',
    'pass_shot_assist': 'count',
    'pass_switch': 'count',
    'pass_through_ball': 'count',
    'shot_aerial_won': 'count',
    'shot_first_time': 'count',
    'shot_key_pass_id': 'count',
    'shot_one_on_one': 'count',
    'shot_open_goal': 'count',
    'shot_statsbomb_xg': 'sum',
    'under_pressure': 'sum'
}

# Iterar sobre cada match_id
for match_id in matches['match_id']:
    # Obtener eventos del partido
    events = sb.events(match_id=match_id)
    
    # Verificar las columnas disponibles
    available_columns = set(events.columns)
    valid_agg_methods = {col: method for col, method in columns_agg_methods.items() if col in available_columns}
    
    # Asegurarse de que 'player' y 'player_id' estén en el DataFrame
    required_columns = ['player', 'player_id']
    for col in required_columns:
        if col not in valid_agg_methods:
            valid_agg_methods[col] = 'first'
    
    # Seleccionar las columnas necesarias
    selected_events = events[list(valid_agg_methods.keys())]
    
    # Agrupar los eventos por jugador
    grouped_events = selected_events.groupby(['player', 'player_id'], as_index=False).agg(valid_agg_methods)

    # Añadir el DataFrame del partido a la lista
    grouped_events_list.append(grouped_events)

# Concatenar todos los DataFrames de los partidos en uno solo
all_grouped_events = pd.concat(grouped_events_list, ignore_index=True)

# Agrupar nuevamente por jugador para sumar los valores agregados de todos los partidos
final_grouped_events = all_grouped_events.groupby(['player', 'player_id', 'position'], as_index=False).sum()

# Muestra el DataFrame final
print(final_grouped_events)

# Guardar el DataFrame final en un archivo CSV
file_path = r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\LaLiga_20_21.csv'
final_grouped_events.to_csv(file_path, index=False)

