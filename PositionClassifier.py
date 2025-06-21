# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:57:31 2024

@author: Sebastián Serra
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Leer el archivo CSV
file_path = r'D:\AZTEC DH  COLLABORATIVE\Scouting & Appraisal\LaLiga_20_21.csv'
df = pd.read_csv(file_path)

# Reemplazar las posiciones según las especificaciones
df = df[df['position'] != 'Deep Winger']
df['position'] = df['position'].replace({
    'Left Center Forward': 'Center Forward',
    'Right Center Forward': 'Center Forward',
    'Left Back': 'Fullback',
    'Right Back': 'Fullback',
    'Left Wing Back': 'Deep Winger',
    'Right Wing Back': 'Deep Winger',
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
    'Left Attacking Midfield': None,
    'Right Attacking Midfield': None
})

# Quitar las posiciones que son None
df = df.dropna(subset=['position'])

# Generar una lista con todas las diferentes posiciones
positions = df['position'].unique().tolist()

# Mostrar la lista de posiciones
print("Lista de posiciones:", positions)

# Definir las posiciones por categoría
defense_positions = ['Goalkeeper', 'Fullback', 'Center Back', 'Center Defensive Midfield']
midfield_positions = ['Center Midfield', 'Wide Midfield']
attack_positions = ['Center Forward', 'Winger']

# Dividir las posiciones en sublistas
defense = [pos for pos in positions if pos in defense_positions]
midfield = [pos for pos in positions if pos in midfield_positions]
attack = [pos for pos in positions if pos in attack_positions]

# Filtrar el DataFrame por posición
df_defense = df[df['position'].isin(defense_positions)]
df_midfield = df[df['position'].isin(midfield_positions)]
df_attack = df[df['position'].isin(attack_positions)]

# Función para entrenar y evaluar un modelo de clasificación
def train_and_evaluate_model(df, label_column='position', title='Confusion Matrix'):
    # Preparar los datos
    X = df.drop(columns=[label_column])
    X = X.drop(columns = ['player', 'player_id'])
    y = df[label_column]

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Sobremuestreo para clases desequilibradas
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)

    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    # Normalizar la matriz de confusión
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear un DataFrame para la visualización
    cm_df = pd.DataFrame(cm_normalized, index=model.classes_, columns=model.classes_)
    
    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='.2%', cmap='Blues', cbar=False, 
                linewidths=.5, linecolor='gray')
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    # Imprimir el reporte de clasificación con recall
    print(f"Reporte de clasificación para {title}:")
    print(classification_report(y_test, y_pred, target_names=model.classes_, zero_division=0))

    return model    
    
# Entrenar modelos para cada sub-DataFrame
print("Entrenando y evaluando modelo para defensa...")
model_defense = train_and_evaluate_model(df_defense, title='Defensa')

print("Entrenando y evaluando modelo para mediocampo...")
model_midfield = train_and_evaluate_model(df_midfield, title='Mediocampo')

print("Entrenando y evaluando modelo para ataque...")
model_attack = train_and_evaluate_model(df_attack, title='Ataque')


# Función para guardar la importancia de las características en un archivo CSV
def save_feature_importance(df, model, title, filename):
    X = df.drop(columns=['position', 'player', 'player_id'])
    features = X.columns
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(filename, index=False)
    print(f"Importancia de características guardada en {filename}")

# Guardar la importancia de características para cada modelo en archivos CSV
save_feature_importance(df_defense, model_defense, 'Defensa', 'feature_importance_defense.csv')
save_feature_importance(df_midfield, model_midfield, 'Mediocampo', 'feature_importance_midfield.csv')
save_feature_importance(df_attack, model_attack, 'Ataque', 'feature_importance_attack.csv')

