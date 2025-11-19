# CML

## Pipeline de Predicción de Temperatura Máxima con RandomForest

Este repositorio contiene una pipeline reproducible para predecir la temperatura máxima (`maxtemp`) usando `RandomForestRegressor` de scikit-learn.

### Instalación

Instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

### Entrenamiento del Modelo

Para entrenar el modelo, ejecuta el siguiente comando:

```bash
python src/train_rf.py --data data.csv
```

**Parámetros disponibles:**
- `--data`: Ruta al archivo CSV de entrada (por defecto: `data.csv`)
- `--model-out`: Ruta de salida para el modelo entrenado (por defecto: `models/rf_pipeline.joblib`)
- `--test-size`: Proporción de datos para test (por defecto: `0.2`)
- `--random-state`: Semilla aleatoria para reproducibilidad (por defecto: `42`)
- `--n-estimators`: Número de árboles en RandomForest (por defecto: `100`)

**Ejemplo con parámetros personalizados:**

```bash
python src/train_rf.py --data data.csv --n-estimators 200 --test-size 0.3
```

El script de entrenamiento:
- Lee el CSV y parsea la columna `Date` con formato `dd.mm.yyyy`
- Crea features temporales: `month`, `day`, `dayofyear`
- Usa features numéricas: `mintemp`, `pressure`, `humidity`, `mean wind speed`, `dayofyear`, `month`, `day`
- Usa features categóricas: `weather`
- Aplica imputación y escalado/encoding según el tipo de feature
- Entrena un modelo RandomForest
- Evalúa el modelo con MAE y R² en el conjunto de test
- Guarda el pipeline completo en `models/rf_pipeline.joblib`

### Predicción

Para realizar predicciones con el modelo entrenado:

```bash
python src/predict_rf.py --input new.csv
```

**Parámetros disponibles:**
- `--model`: Ruta al modelo entrenado (por defecto: `models/rf_pipeline.joblib`)
- `--input`: Ruta al archivo de entrada CSV o JSON (requerido)
- `--output`: Ruta al archivo de salida CSV (por defecto: `predictions.csv`)

**Ejemplo:**

```bash
python src/predict_rf.py --input new_data.csv --output results.csv
```

El script de predicción:
- Carga el pipeline entrenado
- Lee los datos de entrada (soporta CSV y JSON)
- Genera las features temporales si es necesario
- Aplica las transformaciones y genera predicciones
- Guarda los resultados en un archivo CSV con la columna `prediction`

### Notas

- El script de entrenamiento usa `dropna` para eliminar filas con valores faltantes en columnas importantes antes de entrenar
- Durante el entrenamiento y predicción, se aplican imputaciones mediante `SimpleImputer` para valores faltantes
- Si prefieres estrategias de imputación diferentes, puedes modificar los scripts según tus necesidades
- El archivo `data.csv` original no se modifica durante el proceso