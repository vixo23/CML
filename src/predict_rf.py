#!/usr/bin/env python3
"""
Script para hacer predicciones de temperatura máxima usando un modelo entrenado.
"""

import argparse
import sys
import pandas as pd
import numpy as np
import joblib


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Predecir temperatura máxima usando modelo RandomForest entrenado'
    )
    parser.add_argument('--model', type=str, default='models/rf_pipeline.joblib',
                        help='Ruta al modelo guardado (default: models/rf_pipeline.joblib)')
    parser.add_argument('--input', type=str, required=True,
                        help='Ruta al archivo de entrada (CSV o JSON)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Ruta para guardar predicciones (default: predictions.csv)')
    return parser.parse_args()


def load_model(model_path):
    """Cargar el modelo desde disco."""
    try:
        print(f"Cargando modelo desde {model_path}...")
        pipeline = joblib.load(model_path)
        print("Modelo cargado exitosamente!")
        return pipeline
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        sys.exit(1)


def load_input_data(input_path):
    """Cargar datos de entrada desde CSV o JSON."""
    try:
        print(f"Cargando datos desde {input_path}...")
        
        if input_path.endswith('.json'):
            df = pd.read_json(input_path)
        elif input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        else:
            # Intentar CSV por defecto
            df = pd.read_csv(input_path)
        
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        sys.exit(1)


def prepare_features(df):
    """Preparar features para predicción."""
    # Parsear fecha si existe
    if 'Date' in df.columns:
        print("Parseando columna 'Date'...")
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d.%m.%Y', errors='coerce')
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofyear'] = df['Date'].dt.dayofyear
    else:
        # Si no hay columna Date, crear las columnas temporales con valores faltantes
        if 'month' not in df.columns:
            df['month'] = np.nan
        if 'day' not in df.columns:
            df['day'] = np.nan
        if 'dayofyear' not in df.columns:
            df['dayofyear'] = np.nan
    
    # Asegurar que existen todas las columnas requeridas
    required_columns = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                       'weather', 'dayofyear', 'month', 'day']
    
    for col in required_columns:
        if col not in df.columns:
            if col == 'weather':
                df[col] = 'missing'
            else:
                df[col] = np.nan
    
    # Seleccionar solo las columnas necesarias
    X = df[required_columns]
    
    return X, df


def main():
    """Función principal."""
    args = parse_args()
    
    # Cargar modelo
    pipeline = load_model(args.model)
    
    # Cargar datos de entrada
    df = load_input_data(args.input)
    
    # Preparar features
    X, df_original = prepare_features(df)
    
    print(f"\nRealizando predicciones para {X.shape[0]} muestras...")
    
    try:
        # Hacer predicciones
        predictions = pipeline.predict(X)
        print("Predicciones completadas!")
        
        # Añadir predicciones al dataframe original
        df_output = df_original.copy()
        df_output['prediction'] = predictions
        
        # Guardar resultados
        print(f"\nGuardando predicciones en {args.output}...")
        df_output.to_csv(args.output, index=False)
        print("Predicciones guardadas exitosamente!")
        
        # Mostrar resumen
        print(f"\n{'='*50}")
        print(f"RESUMEN DE PREDICCIONES:")
        print(f"{'='*50}")
        print(f"Número de predicciones: {len(predictions)}")
        print(f"Temperatura máxima predicha (media): {predictions.mean():.2f}")
        print(f"Temperatura máxima predicha (min): {predictions.min():.2f}")
        print(f"Temperatura máxima predicha (max): {predictions.max():.2f}")
        print(f"{'='*50}")
        
        print(f"\nPrimeras 5 predicciones:")
        print(df_output[['prediction']].head())
        
    except Exception as e:
        print(f"Error al hacer predicciones: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
