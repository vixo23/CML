# CML - Pipeline de Predicción de Temperatura Máxima

Este proyecto contiene un pipeline reproducible para predecir la temperatura máxima (maxtemp) usando RandomForestRegressor con preprocesamiento automatizado.

## Instalación

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Uso

### 1. Entrenar el Modelo

Para entrenar el modelo RandomForest con los datos:

```bash
python src/train_rf.py --data data.csv
```

Parámetros disponibles:
- `--data`: Ruta al archivo CSV de datos (default: `data.csv`)
- `--model-out`: Ruta para guardar el modelo (default: `models/rf_pipeline.joblib`)
- `--test-size`: Proporción de datos para test (default: `0.2`)
- `--random-state`: Semilla aleatoria para reproducibilidad (default: `42`)
- `--n-estimators`: Número de árboles en RandomForest (default: `100`)

Ejemplo con parámetros personalizados:

```bash
python src/train_rf.py --data data.csv --n-estimators 200 --test-size 0.3
```

El script mostrará las métricas MAE (Error Absoluto Medio) y R² en el conjunto de test.

### 2. Hacer Predicciones

Para hacer predicciones con el modelo entrenado:

```bash
python src/predict_rf.py --input new_data.csv --model models/rf_pipeline.joblib
```

Parámetros disponibles:
- `--model`: Ruta al modelo guardado (default: `models/rf_pipeline.joblib`)
- `--input`: Ruta al archivo de entrada (CSV o JSON) - **requerido**
- `--output`: Ruta para guardar predicciones (default: `predictions.csv`)

Ejemplo:

```bash
python src/predict_rf.py --input new_data.csv --output resultados.csv
```

## Preprocesamiento y Features

El pipeline incluye preprocesamiento automatizado:

### Features Numéricas
- `mintemp`: Temperatura mínima
- `pressure`: Presión atmosférica
- `humidity`: Humedad
- `mean wind speed`: Velocidad media del viento
- `dayofyear`: Día del año (extraído de Date)
- `month`: Mes (extraído de Date)
- `day`: Día del mes (extraído de Date)

**Preprocesamiento**: SimpleImputer (estrategia: mediana) + StandardScaler

### Features Categóricas
- `weather`: Condición meteorológica

**Preprocesamiento**: SimpleImputer (valor constante: 'missing') + OneHotEncoder

### Columna Date
Si el CSV contiene una columna `Date`, se parsea automáticamente (formato: `dd.mm.yyyy`) y se extraen las features temporales (month, day, dayofyear).

## Notas Importantes

- **El archivo data.csv no se modifica**: Todos los cambios se realizan en memoria
- **Manejo de valores faltantes**: El pipeline usa SimpleImputer para manejar NaNs en features
- **Solo se eliminan filas sin target**: Durante el entrenamiento, solo se eliminan filas donde `maxtemp` es NaN
- **Pipeline completo guardado**: El modelo guardado incluye todas las transformaciones de preprocesamiento, por lo que puede aplicarse directamente a datos nuevos
- **El directorio `models/` se crea automáticamente** si no existe

## Estructura del Proyecto

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── train_rf.py      # Script de entrenamiento
│   └── predict_rf.py    # Script de predicción
└── models/              # Modelos entrenados (generado)
    └── rf_pipeline.joblib
```