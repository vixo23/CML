#!/usr/bin/env python3
"""
Script para entrenar un modelo de RandomForestRegressor para predecir temperatura máxima.

Características:
- Carga datos desde un archivo CSV
- Preprocesa la columna de fecha (formato dd.MM.YYYY)
- Crea características temporales (month, day, dayofyear)
- Usa Pipeline con ColumnTransformer para preprocesar features numéricas y categóricas
- Entrena un modelo de RandomForestRegressor
- Evalúa el modelo con MAE y R2
- Guarda el pipeline completo en un archivo joblib
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def parse_date(df, date_column='Date', date_format='%d.%m.%Y'):
    """
    Parsea la columna de fecha y crea características temporales.
    
    Args:
        df: DataFrame con los datos
        date_column: Nombre de la columna de fecha
        date_format: Formato de la fecha
    
    Returns:
        DataFrame con las nuevas características temporales
    """
    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofyear'] = df[date_column].dt.dayofyear
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo RandomForestRegressor para predecir temperatura máxima'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data.csv',
        help='Ruta al archivo CSV con los datos (default: data.csv)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Semilla para reproducibilidad (default: 42)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporción de datos para test (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENTRENAMIENTO DE MODELO RANDOMFOREST PARA PREDICCIÓN DE TEMPERATURA")
    print("=" * 70)
    print()
    
    # 1. Cargar datos
    print(f"1. Cargando datos desde: {args.data}")
    if not os.path.exists(args.data):
        print(f"ERROR: El archivo {args.data} no existe.")
        return 1
    
    df = pd.read_csv(args.data)
    print(f"   - Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"   - Columnas: {list(df.columns)}")
    print()
    
    # 2. Preprocesar fechas
    print("2. Preprocesando fechas y creando características temporales...")
    df = parse_date(df)
    print(f"   - Características temporales creadas: month, day, dayofyear")
    print()
    
    # 3. Eliminar valores nulos
    print("3. Eliminando valores nulos...")
    initial_rows = df.shape[0]
    df = df.dropna()
    final_rows = df.shape[0]
    print(f"   - Filas eliminadas: {initial_rows - final_rows}")
    print(f"   - Filas restantes: {final_rows}")
    print()
    
    # 4. Definir features y target
    print("4. Definiendo features y target...")
    numeric_features = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                        'dayofyear', 'month', 'day']
    categorical_features = ['weather']
    target = 'maxtemp'
    
    X = df[numeric_features + categorical_features]
    y = df[target]
    
    print(f"   - Features numéricas: {numeric_features}")
    print(f"   - Features categóricas: {categorical_features}")
    print(f"   - Target: {target}")
    print()
    
    # 5. Dividir en train y test
    print(f"5. Dividiendo datos (test_size={args.test_size}, random_state={args.random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"   - Train: {X_train.shape[0]} muestras")
    print(f"   - Test: {X_test.shape[0]} muestras")
    print()
    
    # 6. Crear pipeline de preprocesamiento
    print("6. Creando pipeline de preprocesamiento...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Pipeline completo: preprocesamiento + modelo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=args.random_state, n_estimators=100))
    ])
    print(f"   - Pipeline creado con StandardScaler y OneHotEncoder")
    print(f"   - Modelo: RandomForestRegressor (n_estimators=100)")
    print()
    
    # 7. Entrenar modelo
    print("7. Entrenando modelo...")
    pipeline.fit(X_train, y_train)
    print("   - Entrenamiento completado!")
    print()
    
    # 8. Evaluar modelo
    print("8. Evaluando modelo...")
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"   MÉTRICAS DE ENTRENAMIENTO:")
    print(f"   - MAE (train): {train_mae:.4f}")
    print(f"   - R² (train): {train_r2:.4f}")
    print()
    print(f"   MÉTRICAS DE TEST:")
    print(f"   - MAE (test): {test_mae:.4f}")
    print(f"   - R² (test): {test_r2:.4f}")
    print()
    
    # 9. Guardar modelo
    print("9. Guardando modelo...")
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'rf_pipeline.joblib')
    
    joblib.dump(pipeline, model_path)
    print(f"   - Pipeline guardado en: {model_path}")
    print()
    
    print("=" * 70)
    print("¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())
