#!/usr/bin/env python3
"""
Script para entrenar un modelo RandomForestRegressor para predecir la temperatura máxima.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo RandomForest para predecir temperatura máxima'
    )
    parser.add_argument('--data', type=str, default='data.csv',
                        help='Ruta al archivo CSV de datos (default: data.csv)')
    parser.add_argument('--model-out', type=str, default='models/rf_pipeline.joblib',
                        help='Ruta para guardar el modelo (default: models/rf_pipeline.joblib)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporción de datos para test (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Semilla aleatoria (default: 42)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Número de árboles en RandomForest (default: 100)')
    return parser.parse_args()


def load_and_prepare_data(data_path):
    """Cargar y preparar los datos."""
    try:
        print(f"Cargando datos desde {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Parsear fecha si existe
        if 'Date' in df.columns:
            print("Parseando columna 'Date'...")
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d.%m.%Y', errors='coerce')
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['dayofyear'] = df['Date'].dt.dayofyear
            print("Columnas temporales creadas: month, day, dayofyear")
        
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        sys.exit(1)


def create_pipeline(n_estimators, random_state):
    """Crear pipeline de preprocesamiento y modelo."""
    # Features numéricas y categóricas
    numeric_features = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                       'dayofyear', 'month', 'day']
    categorical_features = ['weather']
    
    # Pipeline para features numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para features categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # ColumnTransformer para aplicar transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Pipeline completo con modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=n_estimators, 
                                       random_state=random_state))
    ])
    
    return pipeline


def main():
    """Función principal."""
    args = parse_args()
    
    # Cargar y preparar datos
    df = load_and_prepare_data(args.data)
    
    # Verificar que existe la columna objetivo
    if 'maxtemp' not in df.columns:
        print("Error: La columna 'maxtemp' no existe en los datos")
        sys.exit(1)
    
    # Eliminar filas sin valor objetivo (dropna solo en maxtemp)
    df_clean = df.dropna(subset=['maxtemp'])
    print(f"Filas después de eliminar NaN en 'maxtemp': {df_clean.shape[0]}")
    
    # Separar features y target
    feature_columns = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                      'weather', 'dayofyear', 'month', 'day']
    
    # Verificar que las columnas necesarias existen
    missing_cols = [col for col in feature_columns if col not in df_clean.columns]
    if missing_cols:
        print(f"Advertencia: Columnas faltantes: {missing_cols}")
        # Crear columnas faltantes con NaN
        for col in missing_cols:
            df_clean[col] = np.nan
    
    X = df_clean[feature_columns]
    y = df_clean['maxtemp']
    
    print(f"\nDimensiones de features: {X.shape}")
    print(f"Dimensiones de target: {y.shape}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de test: {X_test.shape[0]} muestras")
    
    # Crear y entrenar pipeline
    print(f"\nEntrenando RandomForestRegressor con {args.n_estimators} árboles...")
    pipeline = create_pipeline(args.n_estimators, args.random_state)
    pipeline.fit(X_train, y_train)
    print("Entrenamiento completado!")
    
    # Evaluar en conjunto de test
    print("\nEvaluando modelo en conjunto de test...")
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"RESULTADOS EN CONJUNTO DE TEST:")
    print(f"{'='*50}")
    print(f"MAE (Error Absoluto Medio): {mae:.4f}")
    print(f"R² (Coeficiente de Determinación): {r2:.4f}")
    print(f"{'='*50}")
    
    # Guardar modelo
    model_dir = os.path.dirname(args.model_out)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"\nDirectorio creado: {model_dir}")
    
    print(f"\nGuardando pipeline en {args.model_out}...")
    joblib.dump(pipeline, args.model_out)
    print("Pipeline guardado exitosamente!")
    print(f"\nPara hacer predicciones, usa:")
    print(f"  python src/predict_rf.py --model {args.model_out} --input <archivo_datos>")


if __name__ == '__main__':
    main()
