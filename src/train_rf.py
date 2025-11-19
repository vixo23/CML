#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script CLI para entrenar modelo RandomForest para predicción de temperatura máxima.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo RandomForest para predicción de temperatura máxima'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data.csv',
        help='Ruta al archivo CSV de entrada (por defecto: data.csv)'
    )
    parser.add_argument(
        '--model-out',
        type=str,
        default='models/rf_pipeline.joblib',
        help='Ruta de salida para el modelo (por defecto: models/rf_pipeline.joblib)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporción de datos para test (por defecto: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Semilla aleatoria (por defecto: 42)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Número de árboles en RandomForest (por defecto: 100)'
    )
    return parser.parse_args()


def load_and_preprocess_data(data_path):
    """
    Cargar y preprocesar datos.
    
    Args:
        data_path: Ruta al archivo CSV
        
    Returns:
        DataFrame preprocesado
    """
    print(f"Cargando datos desde {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")
    
    # Leer CSV con encoding utf-8
    df = pd.read_csv(data_path, encoding='utf-8')
    
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Parsear columna Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(
            df['Date'], 
            dayfirst=True, 
            format='%d.%m.%Y', 
            errors='coerce'
        )
        
        # Crear features temporales
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofyear'] = df['Date'].dt.dayofyear
        
        print("Features temporales creadas: month, day, dayofyear")
    else:
        print("Advertencia: No se encontró columna 'Date', se omiten features temporales")
    
    return df


def prepare_features_and_target(df):
    """
    Preparar features y variable objetivo.
    
    Args:
        df: DataFrame preprocesado
        
    Returns:
        X, y: Features y target
    """
    # Definir columnas de features
    numeric_features = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                        'dayofyear', 'month', 'day']
    categorical_features = ['weather']
    
    # Filtrar solo las columnas que existen en el DataFrame
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    all_features = numeric_features + categorical_features
    
    print(f"Features numéricas: {numeric_features}")
    print(f"Features categóricas: {categorical_features}")
    
    # Verificar que existe la columna target
    if 'maxtemp' not in df.columns:
        raise ValueError("No se encontró la columna 'maxtemp' (variable objetivo)")
    
    # Eliminar filas con valores faltantes en columnas importantes
    columns_to_check = all_features + ['maxtemp']
    df_clean = df.dropna(subset=columns_to_check)
    
    print(f"Filas después de eliminar valores faltantes: {df_clean.shape[0]}")
    
    if df_clean.shape[0] == 0:
        raise ValueError("No quedan datos después de eliminar valores faltantes")
    
    X = df_clean[all_features]
    y = df_clean['maxtemp']
    
    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features, n_estimators, random_state):
    """
    Construir pipeline de preprocesamiento y modelo.
    
    Args:
        numeric_features: Lista de features numéricas
        categorical_features: Lista de features categóricas
        n_estimators: Número de árboles para RandomForest
        random_state: Semilla aleatoria
        
    Returns:
        Pipeline completo
    """
    # Pipeline para features numéricas
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para features categóricas
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])
    
    # Pipeline final
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        ))
    ])
    
    return pipeline


def train_and_evaluate(pipeline, X, y, test_size, random_state):
    """
    Entrenar y evaluar el modelo.
    
    Args:
        pipeline: Pipeline de sklearn
        X: Features
        y: Target
        test_size: Proporción de test
        random_state: Semilla aleatoria
        
    Returns:
        Pipeline entrenado, métricas
    """
    print(f"\nDividiendo datos: train {1-test_size:.0%}, test {test_size:.0%}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Tamaño train: {X_train.shape[0]} muestras")
    print(f"Tamaño test: {X_test.shape[0]} muestras")
    
    print("\nEntrenando modelo RandomForest...")
    pipeline.fit(X_train, y_train)
    
    print("Prediciendo en conjunto de test...")
    y_pred = pipeline.predict(X_test)
    
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("RESULTADOS EN TEST SET")
    print("="*50)
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50)
    
    return pipeline, {'mae': mae, 'r2': r2}


def save_model(pipeline, output_path):
    """
    Guardar pipeline entrenado.
    
    Args:
        pipeline: Pipeline de sklearn
        output_path: Ruta de salida
    """
    # Crear directorio si no existe
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio creado: {output_dir}")
    
    print(f"\nGuardando modelo en {output_path}...")
    joblib.dump(pipeline, output_path)
    print("Modelo guardado exitosamente!")


def main():
    """Función principal."""
    try:
        # Parsear argumentos
        args = parse_args()
        
        print("="*50)
        print("ENTRENAMIENTO DE MODELO RANDOMFOREST")
        print("="*50)
        print(f"Parámetros:")
        print(f"  - Data: {args.data}")
        print(f"  - Model output: {args.model_out}")
        print(f"  - Test size: {args.test_size}")
        print(f"  - Random state: {args.random_state}")
        print(f"  - N estimators: {args.n_estimators}")
        print("="*50 + "\n")
        
        # Cargar y preprocesar datos
        df = load_and_preprocess_data(args.data)
        
        # Preparar features y target
        X, y, numeric_features, categorical_features = prepare_features_and_target(df)
        
        # Construir pipeline
        pipeline = build_pipeline(
            numeric_features, 
            categorical_features,
            args.n_estimators,
            args.random_state
        )
        
        # Entrenar y evaluar
        trained_pipeline, metrics = train_and_evaluate(
            pipeline, X, y, args.test_size, args.random_state
        )
        
        # Guardar modelo
        save_model(trained_pipeline, args.model_out)
        
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
