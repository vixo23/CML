# CML - Pipeline de Predicción de Temperatura con RandomForest

Este proyecto implementa un pipeline reproducible para predecir la temperatura máxima usando RandomForestRegressor.

## Características

- **Entrenamiento**: Script automatizado para entrenar un modelo de RandomForest con preprocesamiento integrado
- **Predicción**: Script para realizar predicciones sobre nuevos datos (CSV o JSON)
- **Reproducibilidad**: Pipeline completo guardado con joblib, incluyendo preprocesamiento
- **Características temporales**: Extracción automática de features temporales (mes, día, día del año)
- **Preprocesamiento**: StandardScaler para features numéricas y OneHotEncoder para categóricas

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/vixo23/CML.git
cd CML
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
CML/
├── data.csv                    # Datos de entrenamiento
├── src/
│   ├── train_rf.py            # Script de entrenamiento
│   └── predict_rf.py          # Script de predicción
├── models/
│   └── rf_pipeline.joblib     # Pipeline entrenado (generado)
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Este archivo
```

## Formato de Datos

El archivo `data.csv` debe contener las siguientes columnas:

- **Date**: Fecha en formato `dd.MM.YYYY` (ej: 01.01.2023)
- **mintemp**: Temperatura mínima (float)
- **maxtemp**: Temperatura máxima (float) - **TARGET**
- **pressure**: Presión atmosférica (float)
- **humidity**: Humedad (float)
- **mean wind speed**: Velocidad media del viento (float)
- **weather**: Condición climática (string: Sunny, Cloudy, Rainy, etc.)

## Uso

### 1. Entrenar el Modelo

Para entrenar el modelo con los datos por defecto:

```bash
python src/train_rf.py
```

Opciones disponibles:

```bash
python src/train_rf.py --data data.csv --random-state 42 --test-size 0.2
```

**Parámetros:**
- `--data`: Ruta al archivo CSV con los datos (default: `data.csv`)
- `--random-state`: Semilla para reproducibilidad (default: `42`)
- `--test-size`: Proporción de datos para test (default: `0.2`)

**Salida:**
- Modelo guardado en: `models/rf_pipeline.joblib`
- Métricas de evaluación: MAE y R² en train y test
- Información detallada del proceso en consola

**Ejemplo de salida:**
```
======================================================================
ENTRENAMIENTO DE MODELO RANDOMFOREST PARA PREDICCIÓN DE TEMPERATURA
======================================================================

1. Cargando datos desde: data.csv
   - Datos cargados: 59 filas, 7 columnas

...

   MÉTRICAS DE TEST:
   - MAE (test): 0.5432
   - R² (test): 0.9234

¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!
```

### 2. Realizar Predicciones

Para realizar predicciones sobre nuevos datos:

#### Entrada CSV:

```bash
python src/predict_rf.py --input datos_nuevos.csv --output predicciones.csv
```

#### Entrada JSON:

```bash
python src/predict_rf.py --input datos_nuevos.json --output predicciones.csv
```

**Parámetros:**
- `--model`: Ruta al modelo entrenado (default: `models/rf_pipeline.joblib`)
- `--input`: Ruta al archivo de entrada (CSV o JSON) - **REQUERIDO**
- `--output`: Ruta al archivo de salida (CSV) - **REQUERIDO**

**Formato del archivo de entrada:**

El archivo de entrada debe contener las mismas columnas que `data.csv` (excepto `maxtemp`):

CSV:
```csv
Date,mintemp,pressure,humidity,mean wind speed,weather
01.03.2023,15.5,1013.2,65,12.3,Sunny
02.03.2023,16.2,1012.8,63,11.5,Cloudy
```

JSON:
```json
[
  {
    "Date": "01.03.2023",
    "mintemp": 15.5,
    "pressure": 1013.2,
    "humidity": 65,
    "mean wind speed": 12.3,
    "weather": "Sunny"
  },
  {
    "Date": "02.03.2023",
    "mintemp": 16.2,
    "pressure": 1012.8,
    "humidity": 63,
    "mean wind speed": 11.5,
    "weather": "Cloudy"
  }
]
```

**Salida:**

El archivo de salida contendrá todas las columnas del archivo de entrada más una columna adicional `prediction` con la temperatura máxima predicha.

### 3. Pipeline Completo

Ejemplo de flujo de trabajo completo:

```bash
# 1. Entrenar el modelo
python src/train_rf.py --data data.csv --random-state 42 --test-size 0.2

# 2. Realizar predicciones
python src/predict_rf.py --input nuevos_datos.csv --output resultados.csv

# 3. Ver los resultados
head resultados.csv
```

## Preprocesamiento

El pipeline aplica automáticamente los siguientes pasos:

1. **Parseo de fechas**: Conversión de fecha en formato `dd.MM.YYYY`
2. **Características temporales**: Creación de `month`, `day`, `dayofyear`
3. **Eliminación de valores nulos**: Se eliminan las filas con valores faltantes
4. **Estandarización**: StandardScaler para features numéricas
5. **Codificación**: OneHotEncoder para features categóricas

## Notas Importantes

- **Valores faltantes**: El script de entrenamiento elimina automáticamente las filas con valores nulos
- **Reproducibilidad**: Usar el mismo `random-state` garantiza resultados reproducibles
- **Formato de fecha**: La fecha debe estar en formato `dd.MM.YYYY` (día.mes.año)
- **Columna weather**: Debe ser categórica (ej: Sunny, Cloudy, Rainy)
- **Pipeline guardado**: El modelo incluye todo el preprocesamiento, no es necesario aplicarlo manualmente al predecir

## Dependencias

- pandas
- numpy
- scikit-learn
- joblib
- matplotlib (opcional, para visualización)
- seaborn (opcional, para visualización)

## Troubleshooting

**Error: "El modelo no existe"**
```bash
# Solución: Entrenar el modelo primero
python src/train_rf.py
```

**Error: "Faltan columnas requeridas"**
```bash
# Solución: Verificar que el archivo de entrada tenga todas las columnas necesarias
# Date, mintemp, pressure, humidity, mean wind speed, weather
```

**Error: "Error al preprocesar fechas"**
```bash
# Solución: Verificar que la columna Date esté en formato dd.MM.YYYY
# Ejemplo correcto: 01.01.2023
# Ejemplo incorrecto: 2023-01-01
```

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerencias o mejoras.

## Licencia

Este proyecto está bajo la licencia MIT.