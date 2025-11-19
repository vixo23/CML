#!/usr/bin/env python3
"""
Script para realizar predicciones usando un modelo RandomForest entrenado.

Características:
- Carga un modelo previamente entrenado desde un archivo joblib
- Acepta datos de entrada en formato CSV o JSON
- Aplica el mismo preprocesamiento que el pipeline guardado
- Genera predicciones y las guarda en un archivo CSV
"""

import argparse
import os
import pandas as pd
import json
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


def load_data(input_path):
    """
    Carga datos desde un archivo CSV o JSON.
    
    Args:
        input_path: Ruta al archivo de entrada
    
    Returns:
        DataFrame con los datos
    """
    if input_path.endswith('.csv'):
        return pd.read_csv(input_path)
    elif input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            data = json.load(f)
        # Si es una lista de diccionarios
        if isinstance(data, list):
            return pd.DataFrame(data)
        # Si es un diccionario con una clave que contiene los datos
        elif isinstance(data, dict):
            # Intentar encontrar la clave con los datos
            if 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                # Asumir que es un solo registro
                return pd.DataFrame([data])
    else:
        raise ValueError(f"Formato de archivo no soportado: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Realizar predicciones con modelo RandomForest entrenado'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/rf_pipeline.joblib',
        help='Ruta al modelo entrenado (default: models/rf_pipeline.joblib)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Ruta al archivo de entrada (CSV o JSON)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Ruta al archivo de salida (CSV con columna prediction)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PREDICCIÓN CON MODELO RANDOMFOREST")
    print("=" * 70)
    print()
    
    # 1. Cargar modelo
    print(f"1. Cargando modelo desde: {args.model}")
    if not os.path.exists(args.model):
        print(f"ERROR: El modelo {args.model} no existe.")
        print("   Por favor, entrena primero el modelo usando train_rf.py")
        return 1
    
    pipeline = joblib.load(args.model)
    print("   - Modelo cargado correctamente!")
    print()
    
    # 2. Cargar datos de entrada
    print(f"2. Cargando datos de entrada desde: {args.input}")
    if not os.path.exists(args.input):
        print(f"ERROR: El archivo {args.input} no existe.")
        return 1
    
    try:
        df = load_data(args.input)
        print(f"   - Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"   - Columnas: {list(df.columns)}")
    except Exception as e:
        print(f"ERROR al cargar datos: {e}")
        return 1
    print()
    
    # 3. Preprocesar fechas
    print("3. Preprocesando fechas y creando características temporales...")
    try:
        df = parse_date(df)
        print(f"   - Características temporales creadas: month, day, dayofyear")
    except Exception as e:
        print(f"ERROR al preprocesar fechas: {e}")
        print("   Asegúrate de que existe una columna 'Date' en formato dd.MM.YYYY")
        return 1
    print()
    
    # 4. Preparar features
    print("4. Preparando features para predicción...")
    numeric_features = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 
                        'dayofyear', 'month', 'day']
    categorical_features = ['weather']
    
    required_columns = numeric_features + categorical_features
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"ERROR: Faltan columnas requeridas: {missing_columns}")
        return 1
    
    X = df[numeric_features + categorical_features]
    print(f"   - Features preparadas: {X.shape[1]} columnas")
    print()
    
    # 5. Realizar predicciones
    print("5. Realizando predicciones...")
    try:
        predictions = pipeline.predict(X)
        df['prediction'] = predictions
        print(f"   - Predicciones realizadas: {len(predictions)} valores")
        print(f"   - Rango de predicciones: [{predictions.min():.2f}, {predictions.max():.2f}]")
    except Exception as e:
        print(f"ERROR al realizar predicciones: {e}")
        return 1
    print()
    
    # 6. Guardar resultados
    print(f"6. Guardando resultados en: {args.output}")
    try:
        # Crear directorio si no existe
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(args.output, index=False)
        print(f"   - Resultados guardados correctamente!")
        print(f"   - Columnas en el archivo: {list(df.columns)}")
    except Exception as e:
        print(f"ERROR al guardar resultados: {e}")
        return 1
    print()
    
    print("=" * 70)
    print("¡PREDICCIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())
