import pandas as pd
import numpy as np
from functions.cleaning import format_data
from tensorflow.keras.models import load_model

def predict_raw_version_a(df, bundle, model):
    """
    Realiza la predicción del precio de autos usados usando un modelo
    que requiere escalado de variables y codificación mixta
    (target encoding + one-hot encoding).

    Esta versión está pensada para modelos como:
    - Regresión Lineal
    - MLP
    - Modelos que usan escaladores en X y y

    Args:
        df (pandas.DataFrame):
            DataFrame con los datos crudos del auto a predecir.
            Puede contener errores de formato, valores faltantes
            o columnas incompletas.

        bundle (dict):
            Diccionario con los artefactos del entrenamiento, incluyendo:
                - training_medians
                - encoding
                - global_mean
                - features
                - x_scaler
                - y_scaler
                - categorical_columns
                - numerical_columns

        model: (estimator)
            Modelo entrenado con método predict().

    Returns
        numpy.ndarray
            Vector con la predicción del precio en euros.
    """
    # Formateo de los datos
    df = format_data(df).copy()

    # Imputación: si algo quedó NaN tras format_data, se usa la media del entrenamiento
    for col, value in bundle["training_medians"].items():
        if col in df.columns:
            # Si la columna existe, rellenar NaNs
            df[col] = df[col].fillna(value)
        else:
            # Si no existe, crear la columna y ponerle la mediana
            df[col] = value

    # Cargar assets del bundle
    encoding = bundle["encoding"]
    global_mean = bundle["global_mean"]
    features = bundle["features"]
    x_scaler = bundle["x_scaler"]
    categorical_columns = bundle["categorical_columns"]
    numerical_columns = bundle["numerical_columns"]
    y_scaler = bundle["y_scaler"]

    # Target encoding
    if "model" in df.columns:
        df["model_encoded"] = df["model"].map(encoding).fillna(global_mean)
        df = df.drop(columns=["model"])
    else:
        # Si no viene la columna de "model", se asume el valor promedio
        df["model_encoded"] = global_mean

    # One-Hot encoding
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown")
        
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Alinear columnas
    df = df.reindex(columns=features, fill_value=0)

    # Convertir booleanos a enteros (0/1)
    bools_cols = df.select_dtypes(include="bool").columns
    df[bools_cols] = df[bools_cols].astype(int)

    # Escalado de X
    df[numerical_columns] = x_scaler.transform(df[numerical_columns])

    # Predicción escalada
    y_pred_scaled = model.predict(df)

    # Desescalar predicción (forzando a que sea una columna)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    return y_pred.flatten()


def predict_raw_version_b(df, bundle, model):
    """
    Predice el precio de autos usados usando modelos que trabajan
    directamente sobre variables numéricas codificadas y, opcionalmente,
    sobre el target transformado con logaritmo.

    Esta versión está diseñada para modelos como:
    - Random Forest
    - XGBoost
    - LightGBM
    - Decision Trees

    Args:
        df (pandas.DataFrame): DataFrame con los datos crudos del auto a predecir.

        bundle (dict):
            Diccionario con los artefactos del entrenamiento, incluyendo:
            - training_medians
            - maps_encoding
            - global_mean
            - features
            - log_transformed

        model (sklearn-like estimator): Modelo entrenado con método predict().

    Returns
        numpy.ndarray
            Vector con la predicción del precio en euros.
    """
    # Formateo de los datos
    df = format_data(df)

    # Imputación: si algo quedó NaN tras format_data, se usa la media del entrenamiento
    for col, value in bundle["training_medians"].items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
        else:
            # Si la columna numérica no existe, crearla con la respectiva mediana
            df[col] = value

    # Cargar assets del bundle
    maps_encoding = bundle["maps_encoding"]
    global_mean = bundle["global_mean"]
    features = bundle["features"]
    log_transformed = bundle["log_transformed"]

    # Target encoding
    for col, mapping in maps_encoding.items():
        if col in df.columns:
            df[col + "_encoded"] = df[col].map(mapping).fillna(global_mean)
        else: 
            # Falta columna, entonces se le asigna el valor promedio del mercado
            df[col+"_encoded"] = global_mean

    # Eliminar columnas categóricas originales
    df = df.drop(columns=maps_encoding.keys(), errors="ignore")

    # Alinear columnas
    df = df.reindex(columns=features, fill_value=0)

    # Convertir todo a float
    df = df.astype(float)

    # Predicción
    y_pred = model.predict(df)

    # Inversa log
    if log_transformed:
        y_pred = np.expm1(y_pred)

    return y_pred.flatten()


def predict_raw_ensamble(df, bundle):
    """
    Realiza la predicción del precio usando un modelo en ensamble
    tipo stacking, combinando múltiples modelos base y un meta-modelo.

    El flujo es:
    1. Limpieza y formateo de datos
    2. Codificación por target encoding
    3. Predicción con modelos base
    4. Predicción final con el meta-modelo

    Args:
        df (pandas.DataFrame):
            DataFrame con los datos crudos del auto a predecir.

        bundle (dict):
            Diccionario que contiene:
                - training_medians
                - maps_encoding
                - global_mean
                - features
                - base_model_rf
                - base_model_xgb
                - meta_model
                - log_transformed

    Returns
        numpy.ndarray
            Vector con la predicción final del precio en euros.
    """
    # Formateo
    df = format_data(df)

    # Imputación
    for col, value in bundle["training_medians"].items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
        else:
            # Si la columna numérica no existe, crearla con la respectiva mediana
            df[col] = value

    # Cargar assets del bundle
    maps_encoding = bundle["maps_encoding"]
    global_mean = bundle["global_mean"]
    features = bundle["features"]
    log_transformed = bundle["log_transformed"]

    # Target encoding
    for col, mapping in maps_encoding.items():
        if col in df.columns:
            df[col + "_encoded"] = df[col].map(mapping).fillna(global_mean)
        else: 
            # Falta columna, entonces se le asigna el valor promedio del mercado
            df[col+"_encoded"] = global_mean

    # Eliminar columnas categóricas originales
    df = df.drop(columns=maps_encoding.keys(), errors="ignore")

    # Alinear columnas
    df = df.reindex(columns=features, fill_value=0)

    # Convertir todo a float
    df = df.astype(float)

    # Cargar los modelos del bundle
    rf_model = bundle["base_model_rf"]
    xgb_model = bundle["base_model_xgb"]
    meta_model = bundle["meta_model"]

    # Predicciones base
    preds_rf = rf_model.predict(df)
    preds_xgb = xgb_model.predict(df)

    # Stacking
    X_meta = np.column_stack([preds_rf, preds_xgb])

    # Predicción final
    y_pred = meta_model.predict(X_meta)

    # Inversa log
    if log_transformed:
        y_pred = np.expm1(y_pred)

    return y_pred.flatten()






