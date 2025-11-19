#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script CLI para realizar predicciones con el modelo RandomForest entrenado.
"""

import argparse
import os
import sys
import json
import pandas as pd
import joblib


def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Realizar predicciones con modelo RandomForest entrenado'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/rf_pipeline.joblib',
        help='Ruta al modelo entrenado (por defecto: models/rf_pipeline.joblib)'
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
        default='predictions.csv',
        help='Ruta al archivo de salida CSV (por defecto: predictions.csv)'
    )
    return parser.parse_args()


def load_model(model_path):
    """
    Cargar pipeline entrenado.
    
    Args:
        model_path: Ruta al modelo
        
    Returns:
        Pipeline cargado
    """
    print(f"Cargando modelo desde {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    pipeline = joblib.load(model_path)
    print("Modelo cargado exitosamente!")
    
    return pipeline


def load_input_data(input_path):
    """
    Cargar datos de entrada desde CSV o JSON.
    
    Args:
        input_path: Ruta al archivo de entrada
        
    Returns:
        DataFrame con los datos
    """
    print(f"Cargando datos de entrada desde {input_path}...")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")
    
    # Determinar formato por extensión
    if input_path.endswith('.json'):
        # Cargar JSON (list of records)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Datos JSON cargados: {df.shape[0]} registros")
    elif input_path.endswith('.csv'):
        # Cargar CSV
        df = pd.read_csv(input_path, encoding='utf-8')
        print(f"Datos CSV cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    else:
        raise ValueError(
            f"Formato de archivo no soportado: {input_path}. "
            "Use .csv o .json"
        )
    
    return df


def preprocess_for_prediction(df):
    """
    Preprocesar datos para predicción (generar features temporales si es necesario).
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame preprocesado
    """
    df = df.copy()
    
    # Parsear columna Date si existe
    if 'Date' in df.columns:
        print("Procesando columna 'Date'...")
        df['Date'] = pd.to_datetime(
            df['Date'], 
            dayfirst=True, 
            format='%d.%m.%Y', 
            errors='coerce'
        )
        
        # Crear features temporales si no existen
        if 'month' not in df.columns:
            df['month'] = df['Date'].dt.month
        if 'day' not in df.columns:
            df['day'] = df['Date'].dt.day
        if 'dayofyear' not in df.columns:
            df['dayofyear'] = df['Date'].dt.dayofyear
        
        print("Features temporales generadas")
    
    return df


def make_predictions(pipeline, df):
    """
    Realizar predicciones con el pipeline.
    
    Args:
        pipeline: Pipeline entrenado
        df: DataFrame con los datos
        
    Returns:
        DataFrame con predicciones
    """
    print("\nRealizando predicciones...")
    
    # Obtener las features que el pipeline espera
    # El pipeline debería manejar las columnas que necesita
    try:
        predictions = pipeline.predict(df)
        print(f"Predicciones generadas: {len(predictions)} valores")
    except Exception as e:
        print(f"Error al predecir: {str(e)}")
        print("\nAsegúrese de que el archivo de entrada contiene las columnas necesarias:")
        print("  - mintemp, pressure, humidity, mean wind speed, weather")
        print("  - Date (opcional, para generar features temporales)")
        raise
    
    # Añadir predicciones al dataframe
    result_df = df.copy()
    result_df['prediction'] = predictions
    
    return result_df


def save_predictions(df, output_path):
    """
    Guardar predicciones a CSV.
    
    Args:
        df: DataFrame con predicciones
        output_path: Ruta de salida
    """
    print(f"\nGuardando predicciones en {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print("Predicciones guardadas exitosamente!")
    
    # Mostrar resumen
    print("\n" + "="*50)
    print("RESUMEN DE PREDICCIONES")
    print("="*50)
    print(f"Total de predicciones: {len(df)}")
    print(f"Temperatura máxima predicha (promedio): {df['prediction'].mean():.2f}")
    print(f"Temperatura máxima predicha (mín): {df['prediction'].min():.2f}")
    print(f"Temperatura máxima predicha (máx): {df['prediction'].max():.2f}")
    print("="*50)
    
    # Mostrar primeras predicciones
    print("\nPrimeras predicciones:")
    print(df[['prediction']].head(10).to_string())


def main():
    """Función principal."""
    try:
        # Parsear argumentos
        args = parse_args()
        
        print("="*50)
        print("PREDICCIÓN CON MODELO RANDOMFOREST")
        print("="*50)
        print(f"Parámetros:")
        print(f"  - Model: {args.model}")
        print(f"  - Input: {args.input}")
        print(f"  - Output: {args.output}")
        print("="*50 + "\n")
        
        # Cargar modelo
        pipeline = load_model(args.model)
        
        # Cargar datos de entrada
        df = load_input_data(args.input)
        
        # Preprocesar datos
        df_processed = preprocess_for_prediction(df)
        
        # Realizar predicciones
        result_df = make_predictions(pipeline, df_processed)
        
        # Guardar resultados
        save_predictions(result_df, args.output)
        
        print("\n¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
