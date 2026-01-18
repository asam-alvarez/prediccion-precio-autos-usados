import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def delete_outliers(path_cleaned_data, path_no_outliers_data):
    """
    Elimina valores atípicos del precio utilizando el método del IQR.

    Carga el dataset limpio desde disco, filtra los registros cuyo
    precio se encuentra fuera del rango intercuartílico permitido
    y guarda el dataset resultante sin outliers en un archivo CSV.

    Args:
        path_cleaned_data (str): Ruta al archivo CSV con los datos limpios.
        path_no_outliers_data (str): Ruta donde se guardará el dataset sin outliers.

    Returns:
        pd.DataFrame: Dataset sin valores atípicos en la variable precio.

    Raises:
        FileNotFoundError: Si el archivo de entrada no existe.
    """
    # Verificación de la existencia del archivo
    if not os.path.exists(path_cleaned_data):
        raise FileNotFoundError(f"No se encontró el archivo en {path_cleaned_data}")
    
    # Cargar el archivo
    df = pd.read_csv(path_cleaned_data)

    # Manejo de valores atípicos utilizando el método del IQR
    q1 = df["price_in_euro"].quantile(0.25)
    q3 = df["price_in_euro"].quantile(0.75)
    iqr = q3 - q1

    lower_limit = max(0, q1 - 1.5 * iqr)
    upper_limit = q3 + 1.5 * iqr

    # Filtrar el conjunto de datos en el rango aceptado
    df = df[(df["price_in_euro"] >= lower_limit) & (df["price_in_euro"] <= upper_limit)]

    # Guardar en un csv
    os.makedirs(os.path.dirname(path_no_outliers_data), exist_ok=True)
    df.to_csv(path_no_outliers_data, index=False)

    return df


def data_version_a(path_data):
    """
    Prepara el dataset para entrenamiento usando una combinación de
    target encoding, one-hot encoding y escalado estándar.

    Esta versión:
    - Divide el dataset en conjuntos de entrenamiento y prueba.
    - Aplica target encoding únicamente a la variable 'model'.
    - Aplica one-hot encoding a variables categóricas seleccionadas.
    - Escala variables numéricas con StandardScaler.
    - Escala la variable objetivo con MinMaxScaler.

    Args:
        path_data (str): Ruta al archivo CSV con los datos preparados.

    Returns:
        tuple: Contiene los siguientes elementos:
            - X_train (pd.DataFrame): Variables de entrenamiento sin escalar.
            - X_train_scaled (pd.DataFrame): Variables de entrenamiento escaladas.
            - X_test_scaled (pd.DataFrame): Variables de prueba escaladas.
            - y_train_scaled (np.ndarray): Variable objetivo de entrenamiento escalada.
            - y_test (pd.Series): Variable objetivo de prueba sin escalar.
            - x_scaler (StandardScaler): Escalador ajustado para las variables X.
            - y_scaler (MinMaxScaler): Escalador ajustado para la variable y.
            - encoding (pd.Series): Mapeo de target encoding para la variable 'model'.
            - global_mean (float): Media global del precio en el conjunto de entrenamiento.
            - features (list): Lista ordenada de variables utilizadas por el modelo.
            - categorical_columns (list): Variables categóricas codificadas con one-hot.
            - numerical_columns (list): Variables numéricas escaladas.

    Raises:
        FileNotFoundError: Si el archivo de entrada no existe.
    """
    # Verificación de la existencia del archivo
    if not os.path.exists(path_data):
        raise FileNotFoundError(f"No se encontró el archivo en {path_data}")
    
    # Cargar el archivo
    df = pd.read_csv(path_data)

    X = df.drop(columns=["price_in_euro"])
    y = df["price_in_euro"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Target encoding
    encoding = X_train.assign(price_in_euro=y_train).groupby("model")["price_in_euro"].mean()

    X_train["model_encoded"] = X_train["model"].map(encoding)
    X_test["model_encoded"] = X_test["model"].map(encoding)

    global_mean = y_train.mean()
    X_train["model_encoded"] = X_train["model_encoded"].fillna(global_mean)
    X_test["model_encoded"] = X_test["model_encoded"].fillna(global_mean)

    X_train = X_train.drop(columns=["model"])
    X_test = X_test.drop(columns=["model"])

    # OneHot encoding
    categorical_columns = [
        "brand", "fuel_type", "color", "transmission_type"
    ]

    X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

    # Alinear columnas
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Guardar el orden de las columnas 
    features = X_train.columns.tolist()

    # Pasar de booleanos a 0/1
    bool_cols = X_train.select_dtypes(include="bool").columns

    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols] = X_test[bool_cols].astype(int)

    # Escalado de variables
    numerical_columns = [
        "power_kw", "fuel_consumption_l_100km", "mileage_in_km", "age", "model_encoded"
    ]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    x_scaler = StandardScaler()
    X_train_scaled[numerical_columns] = x_scaler.fit_transform(X_train_scaled[numerical_columns])
    X_test_scaled[numerical_columns] = x_scaler.transform(X_test_scaled[numerical_columns])

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

    return (X_train, X_train_scaled, X_test_scaled, y_train_scaled, y_test, 
            x_scaler, y_scaler, encoding, global_mean, features,
            categorical_columns, numerical_columns)


def data_version_b(path_data, log_transformed):
    """
    Prepara el dataset utilizando target encoding para todas las
    variables categóricas, con opción de transformación logarítmica
    de la variable objetivo.

    Esta versión:
    - Divide el dataset en conjuntos de entrenamiento y prueba.
    - Aplica target encoding a múltiples variables categóricas.
    - Almacena los mapas de codificación para su uso posterior.
    - Permite aplicar una transformación logarítmica a la variable objetivo.

    Args:
        path_data (str): Ruta al archivo CSV con los datos preparados.
        log_transformed (bool): Indica si se aplica log(1 + y) a la variable objetivo.

    Returns:
        tuple: Contiene los siguientes elementos:
            - X_train (pd.DataFrame): Variables de entrenamiento codificadas.
            - X_test (pd.DataFrame): Variables de prueba codificadas.
            - y_train (pd.Series): Variable objetivo de entrenamiento (opcionalmente transformada).
            - y_test (pd.Series): Variable objetivo de prueba.
            - global_mean (float): Media global del precio en el conjunto de entrenamiento.
            - maps_encoding (dict): Diccionario con los mapas de target encoding por variable.
            - features (list): Lista de variables utilizadas por el modelo.
            - numerical_columns (list): Variables numéricas presentes en el dataset.

    Raises:
        FileNotFoundError: Si el archivo de entrada no existe.
    """
    # Verificación de la existencia del archivo
    if not os.path.exists(path_data):
        raise FileNotFoundError(f"No se encontró el archivo en {path_data}")
    
    # Cargar el archivo
    df = pd.read_csv(path_data)

    X = df.drop(columns=["price_in_euro"])
    y = df["price_in_euro"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    numerical_columns = [
        "power_kw", "fuel_consumption_l_100km", "mileage_in_km", "age"
    ]
    
    # Target encoding
    categorical_columns = [
        "model", "brand", "fuel_type", "color", "transmission_type"
    ]

    global_mean = y_train.mean()

    maps_encoding = {}
    for col in categorical_columns:
        encoding = X_train.assign(price_in_euro=y_train).groupby(col)["price_in_euro"].mean()
        maps_encoding[col] = encoding.to_dict()

        # Mapear
        X_train[col + "_encoded"] = X_train[col].map(encoding)
        X_test[col + "_encoded"] = X_test[col].map(encoding)

        # Llenar nulos con la media global
        X_train[col + "_encoded"] = X_train[col + "_encoded"].fillna(global_mean)
        X_test[col + "_encoded"] = X_test[col + "_encoded"].fillna(global_mean)
    
    # Eliminar columnas que ya fueron codificadas
    X_train = X_train.drop(columns=categorical_columns)
    X_test = X_test.drop(columns=categorical_columns)

    features = X_train.columns.tolist()

    if log_transformed:
        # Escalado de y_train
        y_train = np.log1p(y_train)

    return (X_train, X_test, y_train, y_test, global_mean, 
            maps_encoding, features, numerical_columns)


