# Predicción de precios de autos usados con modelos de Machine Learning

## 1. Descripción del problema

El precio de un auto usado suele ser ambiguo, ya que depende de múltiples factores subjetivos como la percepción del vendedor, el estado aparente del vehículo o la comparación con anuncios similares. Sin embargo, una forma más objetiva de estimar su valor consiste en contrastarlo con los precios reales de mercado de ese mismo modelo y de vehículos comparables.

En este contexto, los modelos de Inteligencia Artificial pueden ayudar a predecir un precio más ajustado a la realidad, aprendiendo patrones a partir de grandes volúmenes de datos históricos del mercado automotriz.

---

## 2. Enfoque de la solución

Este proyecto aborda el problema como una tarea de regresión sobre datos tabulares, entrenando y evaluando múltiples modelos de Machine Learning con el objetivo de identificar aquel que ofrezca el mejor balance entre rendimiento predictivo y características técnicas, como el peso del modelo y el tiempo de inferencia.

Los modelos evaluados fueron:

* Regresión Lineal
* Árbol de Decisión
* Random Forest
* XGBoost
* LightGBM
* Perceptrón Multicapa (MLP)
* MLP con arquitectura residual
* Ensamble (Random Forest + XGBoost)

Para los modelos basados en árboles (Árbol de Decisión, Random Forest, XGBoost, LightGBM y Ensamble) se entrenaron dos versiones:

* una utilizando el precio original,
* y otra aplicando una transformación logarítmica al target (`log(price + 1)`), con el fin de reducir la asimetría y mejorar la estabilidad del entrenamiento.

---

## 3. Dataset

* **Fuente**: Kaggle – Germany Used Cars Dataset 2023
* **Tamaño**: 251,079 registros
* **Periodo**: Vehículos fabricados entre 1995 y 2023

### Variables principales

* Marca, modelo y color
* Año de fabricación y fecha de registro
* Precio en euros
* Potencia (kW y PS)
* Tipo de transmisión
* Tipo de combustible y consumo
* Kilometraje (km)

Durante el preprocesamiento:

* se formatearon las variables,
* se eliminaron valores nulos y filas duplicadas,
* se descartaron columnas redundantes,
* y se removieron valores atípicos del precio.

La gran cantidad de datos disponibles permitió aplicar estas limpiezas sin comprometer la representatividad del dataset.

---

## 4. Estructura del proyecto

```
prediccion_precio_autos_usados
├── data
│   ├── raw
│   │   └── cars.csv
│   └── processed
│       ├── cleaned_data.csv
│       └── no_outliers_data.csv
├── functions
│   ├── cleaning.py
│   ├── preparation.py
│   ├── models.py
│   ├── prediction.py
│   └── __init__.py
├── models
│   ├── *.pkl
│   ├── *.joblib
│   └── *.keras
├── notebooks
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preparation.ipynb
│   ├── 04_models.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_analysis_best_model.ipynb
│   └── 07_test_predictions.ipynb
├── results
│   ├── models_results.csv
│   └── *.png
├── main.py
├── requirements.txt
└── README.md
```

* **data/** contiene los datos crudos y procesados
* **functions/** encapsula en funciones reutilizables el trabajo realizado en los notebooks
* **models/** almacena los modelos entrenados y sus artefactos
* **notebooks/** documenta el análisis exploratorio, entrenamiento y evaluación
* **results/** contiene gráficas y métricas comparativas
* **main.py** es el punto de entrada para realizar predicciones

---

## 5. Entrenamiento de los modelos

El entrenamiento de los modelos se realizó de forma exploratoria y comparativa en notebooks. Posteriormente, cada pipeline de entrenamiento fue encapsulado en funciones, permitiendo reutilizar el proceso y facilitar la carga de modelos ya entrenados sin repetir el entrenamiento completo.

---

## 6. Predicción

Para realizar una predicción, el sistema espera un `pandas.DataFrame` con las columnas correspondientes a las características del vehículo. No obstante, el pipeline es robusto ante datos incompletos o inconsistentes:

* si faltan columnas, se imputan usando la mediana calculada en el entrenamiento,
* si existen valores nulos, estos también se imputan automáticamente,
* los datos categóricos se codifican según el esquema aprendido.

El resultado final es una predicción del precio en euros, incluso cuando la información de entrada es parcial o contiene errores de formato.

## 7. Selección del mejor modelo

Para seleccionar el mejor modelo se dio prioridad a los siguientes criterios:

* **Precisión de predicción** (MAPE y R²)
* **Tamaño del modelo**

Los modelos de Random Forest y Ensamble obtuvieron los mejores valores de MAPE y R²; sin embargo, ambos alcanzaron un tamaño cercano a 1 GB, lo que limita su viabilidad práctica.

Los modelos de XGBoost lograron un rendimiento muy cercano en términos de MAPE y R², pero con un tamaño de apenas unos cuantos MB.
Al comparar ambas variantes, XGBoost con transformación logarítmica del target (`log(price + 1)`) mostró una ligera mejora en estabilidad y generalización.

**Modelo seleccionado**: XGBoost Log

---

## 8. Métricas de evaluación

Se utilizaron las siguientes métricas para evaluar el desempeño de los modelos:

* **MAE (Mean Absolute Error)**: error promedio absoluto.
* **RMSE (Root Mean Squared Error)**: penaliza errores grandes y refleja sensibilidad a outliers.
* **MAPE (%)**: métrica intuitiva que indica el error porcentual promedio.
* **R² (Coeficiente de determinación)**: mide qué tan bien las variables explican la variabilidad del precio.
* **Tiempo de entrenamiento (min)**: relevante para escenarios de reentrenamiento.
* **Tiempo de predicción (seg)**: latencia en inferencia.
* **Tamaño del modelo (MB)**: viabilidad para despliegue, especialmente en aplicaciones ligeras o móviles.

Este conjunto de métricas permite balancear precisión, robustez, eficiencia y portabilidad.

---

## 9. Decisiones de diseño relevantes

### 9.1 Target Encoding vs One-Hot Encoding

Las variables categóricas presentan una alta cardinalidad.
Aplicar One-Hot Encoding de forma indiscriminada incrementaría excesivamente la dimensionalidad.

Se implementaron dos estrategias de procesamiento:

* **Versión A (Regresión Lineal y MLPs)**

  * One-Hot Encoding aplicado únicamente al modelo del vehículo.
  * Target Encoding para el resto de variables categóricas.
  * Esta estrategia aprovecha las fortalezas de las redes neuronales sin disparar el número de variables.

* **Versión B (Modelos basados en árboles)**

  * Uso exclusivo de Target Encoding.
  * Los modelos basados en árboles manejan correctamente este tipo de codificación.

---

### 9.2 Transformación logarítmica del precio

Se aplicó la transformación:

```
log(price + 1)
```

con el objetivo de:

* Reducir la influencia de precios extremadamente altos.
* Estabilizar la varianza.
* Mejorar la capacidad de generalización del modelo.

---

### 9.3 ¿Por qué no solo Deep Learning?

Aunque se exploraron modelos de deep learning, en problemas de regresión con datos tabulares los modelos clásicos suelen ser más eficientes.

Con volúmenes de datos por debajo de varios millones de registros, modelos como Random Forest o XGBoost pueden igualar o superar a redes neuronales profundas en:

* Precisión
* Estabilidad
* Interpretabilidad
* Costo computacional

Este proyecto refleja ese comportamiento.

---

### 9.4 ¿Por qué usar un modelo ensamblado?

El modelo ensamblado combina:

* **XGBoost** (aprendizaje secuencial)
* **Random Forest** (aprendizaje en paralelo)

Esta combinación equivale a considerar la opinión de varios expertos con enfoques distintos.
Aunque el ensamble mostró buen desempeño, su tamaño lo hace menos viable para despliegue.

### 9.5 Ingeniería de la variable edad del vehículo

Además del año de fabricación, se creó explícitamente la variable:

```
age = 2023 - year
```

con el objetivo de representar de forma directa la antigüedad del vehículo, ya que el precio de un auto usado suele correlacionarse más fuertemente con su edad que con el año absoluto de fabricación.

Esta transformación permite que los modelos:

* aprendan patrones temporales de depreciación de forma más clara,
* reduzcan la dependencia del año como valor absoluto,
* y mejoren la estabilidad del aprendizaje, especialmente en modelos no lineales.

La variable `age` se utiliza junto con el resto de características numéricas durante el entrenamiento y la predicción.

### 9.6 Posible redundancia entre marca y modelo

El dataset incluye tanto la marca del vehículo (`brand`) como el modelo o línea (`model`).
En la mayoría de los casos, el nombre del modelo ya implica de forma implícita la marca, ya que los fabricantes no suelen reutilizar nombres de línea entre distintas marcas.

Desde un punto de vista informativo, esto sugiere una posible redundancia parcial de la variable `brand`.

No obstante, se decidió conservar ambas variables por las siguientes razones:

* existen posibles excepciones (submarcas, nombres genéricos o inconsistencias del dataset),
* la codificación utilizada (Target Encoding) no incrementa la dimensionalidad,
* y los modelos basados en árboles pueden aprender a ignorar automáticamente variables con bajo aporte informativo.

De esta forma, el modelo decide de manera implícita si la variable `brand` aporta información adicional más allá de la contenida en `model`.


---

## 10. Limitaciones del proyecto

* El modelo subestima precios de autos de lujo y súper lujo.
* En estos casos, el error pasa de cientos a miles de euros.
* El modelo funciona mejor en autos de gama común y media.

Esto sugiere la necesidad de modelos especializados por segmento o variables adicionales.

---

## 11. Instalación y ejecución

### Requisitos

* **Python 3.10.13**
* Las dependencias se encuentran en `requirements.txt`

### Ejecución

```bash
pip install -r requirements.txt
python main.py
```

El archivo `main.py` carga el modelo entrenado y ejecuta predicciones sobre casos de prueba definidos.

