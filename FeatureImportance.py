# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:03:04 2024

@author: Sebastián Serra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función para leer un archivo CSV y graficar la importancia de las características
def plot_feature_importance(filename, title):
    # Leer el archivo CSV
    importance_df = pd.read_csv(filename)
    
    # Normalizar importancias
    importance_df['Importance'] = (importance_df['Importance'] - importance_df['Importance'].min()) / (importance_df['Importance'].max() - importance_df['Importance'].min())
    
    # Graficar la importancia de las características
    plt.figure(figsize=(10, 7))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar la característica más importante en la parte superior
    plt.show()

# Plotear la importancia de características para cada modelo
plot_feature_importance('feature_importance_defense.csv', 'Importancia de Características - Defensa')
plot_feature_importance('feature_importance_midfield.csv', 'Importancia de Características - Mediocampo')
plot_feature_importance('feature_importance_attack.csv', 'Importancia de Características - Ataque')

# Leer el archivo CSV con los datos de jugadores y posiciones
file_path = r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\LaLiga_20_21.csv'
df = pd.read_csv(file_path)

# Reemplazar las posiciones según las especificaciones
df['position'] = df['position'].replace({
    'Left Center Forward': 'Center Forward',
    'Right Center Forward': 'Center Forward',
    'Left Back': 'Fullback',
    'Right Back': 'Fullback',
    'Deep Winger': 'Fullback',
    'Left Wing Back': 'Fullback',
    'Right Wing Back': 'Fullback',
    'Right Center Back': 'Center Back',
    'Left Center Back': 'Center Back',
    'Left Defensive Midfield': 'Center Defensive Midfield',
    'Right Defensive Midfield': 'Center Defensive Midfield',
    'Left Center Midfield': 'Center Midfield',
    'Right Center Midfield': 'Center Midfield',
    'Left Midfield': 'Wide Midfield',
    'Right Midfield': 'Wide Midfield',
    'Left Wing': 'Winger', 
    'Right Wing': 'Winger',
    'Left Attacking Midfield': 'Center Attacking Midfield',
    'Right Attacking Midfield': 'Center Attacking Midfield'
})

# Filtrar filas con posiciones no válidas
df = df.dropna(subset=['position'])

# Calcular percentiles para cada columna numérica
percentiles_df = df.copy()
numeric_columns = percentiles_df.select_dtypes(include=[np.number]).columns

for col in numeric_columns:
    percentiles_df[f'{col}_percentile'] = percentiles_df[col].rank(pct=True) * 100

# Seleccionar columnas relevantes: 'player', 'position', 'player_id' y percentiles
columns_to_keep = ['player', 'position', 'player_id'] + [f'{col}_percentile' for col in numeric_columns]
percentiles_df = percentiles_df[columns_to_keep]

# Guardar el nuevo DataFrame a un archivo CSV
percentiles_file_path = r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\LaLiga_20_21_with_percentiles.csv'
percentiles_df.to_csv(percentiles_file_path, index=False)

# Leer los DataFrames de feature importance
df_defense_importance = pd.read_csv(r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\feature_importance_defense.csv')
df_midfield_importance = pd.read_csv(r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\feature_importance_midfield.csv')
df_attack_importance = pd.read_csv(r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\feature_importance_attack.csv')

# Normalizar las importancias
def normalize_importance(importance_df):
    importance_df['Importance'] = (importance_df['Importance'] - importance_df['Importance'].min()) / (importance_df['Importance'].max() - importance_df['Importance'].min())
    return importance_df

df_defense_importance = normalize_importance(df_defense_importance)
df_midfield_importance = normalize_importance(df_midfield_importance)
df_attack_importance = normalize_importance(df_attack_importance)

# Definir las posiciones para cada grupo
defense_positions = ['Goalkeeper', 'Fullback', 'Center Back', 'Center Defensive Midfield']
midfield_positions = ['Center Midfield', 'Center Attacking Midfield', 'Wide Midfield']
attack_positions = ['Center Forward', 'Winger']

# Función para calcular el rating por jugador basado en feature importance
def calculate_rating(row, importance_df):
    rating = 0
    for feature_name in importance_df['Feature']:
        if feature_name in row.index:
            a_i = importance_df.loc[importance_df['Feature'] == feature_name, 'Importance'].values
            if len(a_i) > 0:
                a_i = a_i[0]
                rating += a_i * row[feature_name]
    return rating

# Crear una lista para los ratings
ratings = []

# Calcular el rating para cada jugador
for idx, row in df.iterrows():
    position = row['position']
    if position in defense_positions:
        importance_df = df_defense_importance
    elif position in midfield_positions:
        importance_df = df_midfield_importance
    elif position in attack_positions:
        importance_df = df_attack_importance
    else:
        # Si la posición no está en ninguna de las categorías, asignar rating 0 o manejar el caso
        ratings.append(np.nan)
        continue
    
    rating = calculate_rating(row, importance_df)
    ratings.append(rating)

# Agregar la columna de ratings al DataFrame
df['rating'] = ratings

# Guardar el DataFrame con los ratings calculados
ratings_file_path = r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\LaLiga_20_21_with_ratings.csv'
df.to_csv(ratings_file_path, index=False)

ratings = pd.read_csv(ratings_file_path)

# Agrupar por "player" y "player_id" y calcular la primera posición y la media de los ratings
grouped_ratings = ratings.groupby(['player', 'player_id']).agg({'position': 'first', 'rating': 'mean'}).reset_index()

# Guardar el DataFrame agrupado a un archivo CSV
grouped_ratings_file_path = r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\Grouped_Ratings.csv'
grouped_ratings.to_csv(grouped_ratings_file_path, index=False)

# Leer el DataFrame agrupado
grouped_ratings = pd.read_csv(r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\Grouped_Ratings.csv')

# Ordenar por posición y rating
grouped_ratings_sorted = grouped_ratings.sort_values(by=['position', 'rating'], ascending=[True, False])

# Obtener los 5 mejores jugadores por posición
top_5_per_position = grouped_ratings_sorted.groupby('position').head(5)

# Imprimir el listado de los mejores 5 jugadores por posición
for position in top_5_per_position['position'].unique():
    print(f"\n--- Top 5 Jugadores en la Posición: {position} ---")
    position_df = top_5_per_position[top_5_per_position['position'] == position]
    print(position_df[['player', 'rating']].to_string(index=False))
