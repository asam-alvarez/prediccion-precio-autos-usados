from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Add, Activation
import joblib
import os
import time
import numpy as np

def save_model(bundle, path):
    """
    Guarda en disco un bundle de modelo entrenado.

    Args:
        bundle (dict): Diccionario que contiene el modelo entrenado
            junto con sus artefactos de preprocesamiento.
        path (str): Ruta donde se guardará el archivo del modelo.
    """
    joblib.dump(bundle, path)

def train_linear_regression(
    X_train, X_train_scaled, y_train_scaled, x_scaler, y_scaler,
    encoding, global_mean, features,
    categorical_columns, numerical_columns
):
    """
    Entrena un modelo de regresión lineal sobre datos escalados.

    Este modelo utiliza variables previamente codificadas y escaladas,
    y devuelve un bundle que incluye el modelo entrenado junto con
    los escaladores y metadatos necesarios para inferencia futura.

    Args:
        X_train (pd.DataFrame): Variables de entrenamiento sin escalar.
        X_train_scaled (pd.DataFrame): Variables de entrenamiento escaladas.
        y_train_scaled (np.ndarray): Variable objetivo escalada.
        x_scaler (StandardScaler): Escalador ajustado para X.
        y_scaler (MinMaxScaler): Escalador ajustado para y.
        encoding (pd.Series): Target encoding aplicado a la variable modelo.
        global_mean (float): Media global del precio.
        features (list): Lista de variables usadas por el modelo.
        categorical_columns (list): Variables categóricas utilizadas.
        numerical_columns (list): Variables numéricas utilizadas.

    Returns:
        dict: Bundle con el modelo entrenado y sus artefactos.
    """

    model = LinearRegression()

    start_time = time.time()
    model.fit(X_train_scaled, y_train_scaled)
    end_time = time.time()
    
    bundle = {
        "model_type": "sklearn",
        "model": model,
        "model_path": None,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "encoding": encoding,
        "global_mean": global_mean,
        "model_features": features,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "training_time": (end_time - start_time),
        "training_medians": X_train[numerical_columns].median().to_dict()
    }

    return bundle

def train_decision_tree(
    X_train, y_train, global_mean,
    maps_encoding, features, numerical_columns, log_transformed
):
    """
    Entrena un Árbol de Decisión optimizado mediante GridSearchCV.

    Permite aplicar una transformación logarítmica a la variable objetivo
    y devuelve un bundle con el mejor modelo encontrado y los elementos
    necesarios para realizar predicciones posteriores.

    Args:
        X_train (pd.DataFrame): Variables de entrenamiento.
        y_train (pd.Series): Variable objetivo.
        global_mean (float): Media global del precio.
        maps_encoding (dict): Mapas de target encoding por variable.
        features (list): Variables utilizadas por el modelo.
        numerical_columns (list): Variables numéricas del dataset.
        log_transformed (bool): Indica si se aplica log(1 + y).

    Returns:
        dict: Bundle con el modelo entrenado y sus metadatos.
    """

    if log_transformed:
        y_train = np.log1p(y_train)

    param_grid = {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "splitter": ["best", "random"],
        "max_depth": [None],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt", "log2", None]
    }

    model = DecisionTreeRegressor(random_state=123)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=16,
        verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    bundle = {
        "model_type": "sklearn",
        "model": grid_search.best_estimator_,
        "model_path": None,
        "global_mean": global_mean,
        "maps_encoding": maps_encoding,
        "features": features,
        "training_time": (end_time - start_time),
        "training_medians": X_train[numerical_columns].median().to_dict(),
        "log_transformed": log_transformed
    }

    return bundle

def train_random_forest(
    X_train, y_train, global_mean,
    maps_encoding, features, numerical_columns, log_transformed
):
    """
    Entrena un modelo Random Forest optimizado mediante GridSearchCV.

    Aplica búsqueda de hiperparámetros y permite el uso opcional de
    transformación logarítmica en la variable objetivo.

    Args:
        X_train (pd.DataFrame): Variables de entrenamiento.
        y_train (pd.Series): Variable objetivo.
        global_mean (float): Media global del precio.
        maps_encoding (dict): Mapas de target encoding.
        features (list): Variables utilizadas por el modelo.
        numerical_columns (list): Variables numéricas.
        log_transformed (bool): Indica si se aplica log(1 + y).

    Returns:
        dict: Bundle con el mejor modelo Random Forest entrenado.
    """
    
    if log_transformed:
        y_train = np.log1p(y_train)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    model = RandomForestRegressor()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=16,
        verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    bundle = {
        "model_type": "sklearn",
        "model": grid_search.best_estimator_,
        "model_path": None,
        "global_mean": global_mean,
        "maps_encoding": maps_encoding,
        "features": features,
        "training_time": (end_time - start_time),
        "training_medians": X_train[numerical_columns].median().to_dict(),
        "log_transformed": log_transformed
    }

    return bundle

def train_xgboost(
    X_train, y_train, global_mean,
    maps_encoding, features, numerical_columns, log_transformed
):
    """
    Entrena un modelo XGBoost optimizado mediante GridSearchCV.

    Utiliza árboles de gradiente y permite aplicar una transformación
    logarítmica a la variable objetivo.

    Args:
        X_train (pd.DataFrame): Variables de entrenamiento.
        y_train (pd.Series): Variable objetivo.
        global_mean (float): Media global del precio.
        maps_encoding (dict): Mapas de target encoding.
        features (list): Variables utilizadas por el modelo.
        numerical_columns (list): Variables numéricas.
        log_transformed (bool): Indica si se aplica log(1 + y).

    Returns:
        dict: Bundle con el modelo XGBoost entrenado.
    """

    if log_transformed:
        y_train = np.log1p(y_train)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [2, 10],
        "learning_rate": [0.01, 0.03],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "gamma": [0, 1, 10],
        "reg_alpha": [0, 0.5],
        "reg_lambda": [1, 10],
        "tree_method": ["hist"]
    }

    model = XGBRegressor(random_state=123)

    grid_search_xgb = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=16,
        verbose=1
    )

    start_time = time.time()
    grid_search_xgb.fit(X_train, y_train)
    end_time = time.time()

    bundle = {
        "model_type": "sklearn",
        "model": grid_search_xgb.best_estimator_,
        "model_path": None,
        "global_mean": global_mean,
        "maps_encoding": maps_encoding,
        "features": features,
        "training_time": (end_time - start_time),
        "training_medians": X_train[numerical_columns].median().to_dict(),
        "log_transformed": log_transformed
    }

    return bundle

def train_lightgbm(
    X_train, y_train, global_mean,
    maps_encoding, features, numerical_columns, log_transformed
):
    """
    Entrena un modelo LightGBM optimizado mediante GridSearchCV.

    Diseñado para manejar relaciones no lineales complejas y grandes
    volúmenes de datos, con opción de transformación logarítmica del target.

    Args:
        X_train (pd.DataFrame): Variables de entrenamiento.
        y_train (pd.Series): Variable objetivo.
        global_mean (float): Media global del precio.
        maps_encoding (dict): Mapas de target encoding.
        features (list): Variables utilizadas por el modelo.
        numerical_columns (list): Variables numéricas.
        log_transformed (bool): Indica si se aplica log(1 + y).

    Returns:
        dict: Bundle con el modelo LightGBM entrenado.
    """

    if log_transformed:
        y_train = np.log1p(y_train)

    param_grid_lgbm = {
        "n_estimators": [100, 200],
        "max_depth": [3, 10],
        "learning_rate": [0.01, 0.1],
        "num_leaves": [20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.5],
        "reg_lambda": [1, 10]
    }

    lgbm = LGBMRegressor(random_state=123)

    grid_search_lgbm = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid_lgbm,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=16,
        verbose=0
    )

    start_time = time.time()
    grid_search_lgbm.fit(X_train, y_train)
    end_time = time.time()

    bundle = {
        "model_type": "sklearn",
        "model": grid_search_lgbm.best_estimator_,
        "model_path": None,
        "global_mean": global_mean,
        "maps_encoding": maps_encoding,
        "features": features,
        "training_time": (end_time - start_time),
        "training_medians": X_train[numerical_columns].median().to_dict(),
        "log_transformed": log_transformed
    }

    return bundle

def train_mlp(
    input_dim, path_file_mlp,
    X_train, X_train_scaled, y_train_scaled,
    x_scaler, y_scaler, encoding, global_mean,
    features, numerical_columns, categorical_columns
):
    """
    Entrena un Perceptrón Multicapa (MLP) para regresión.

    El modelo se entrena sobre variables escaladas y utiliza callbacks
    para regularización, reducción de tasa de aprendizaje y guardado
    del mejor modelo.

    Args:
        input_dim (int): Dimensión de entrada del modelo.
        path_file_mlp (str): Ruta donde se guardará el mejor modelo.
        X_train (pd.DataFrame): Variables de entrenamiento sin escalar.
        X_train_scaled (pd.DataFrame): Variables de entrenamiento escaladas.
        y_train_scaled (np.ndarray): Variable objetivo escalada.
        x_scaler (StandardScaler): Escalador de variables X.
        y_scaler (MinMaxScaler): Escalador de la variable y.
        encoding (pd.Series): Target encoding aplicado.
        global_mean (float): Media global del precio.
        features (list): Variables utilizadas por el modelo.
        numerical_columns (list): Variables numéricas.
        categorical_columns (list): Variables categóricas.

    Returns:
        dict: Bundle con el modelo MLP entrenado y su historial.
    """
    # Perceptrón Multicapa (MLP) - Dataset A
    def create_mlp_model(input_dim, dropout_rate):
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(16, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        outputs = Dense(1, activation="linear")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer="nadam",
            loss="mae",
            metrics=["mae", "mse"]
        )
        return model

    model_mlp = create_mlp_model(input_dim, dropout_rate=0.1)


    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, min_lr=1e-4, patience=5),
        ModelCheckpoint(
            filepath=path_file_mlp,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    start_time = time.time()

    history_mlp = model_mlp.fit(
        X_train_scaled, y_train_scaled,
        epochs=150, batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    end_time = time.time()

    bundle = {
        "model_type": "keras",
        "model": None,
        "model_path": path_file_mlp,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "encoding": encoding,
        "global_mean": global_mean,
        "features": features,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "training_time": (end_time - start_time),
        "training_medians": X_train[numerical_columns].median().to_dict(),
        "history": history_mlp.history
    }

    return bundle

def train_mlp_residual(
    input_dim, path_file_mlp_residual,
    X_train, X_train_scaled, y_train_scaled,
    x_scaler, y_scaler, encoding, global_mean,
    features, numerical_columns, categorical_columns
):
    """
    Entrena un MLP con arquitectura residual para regresión.

    Utiliza bloques residuales para mejorar el flujo de gradientes
    y la estabilidad del entrenamiento en redes profundas.

    Args:
        input_dim (int): Dimensión de entrada del modelo.
        path_file_mlp_residual (str): Ruta donde se guardará el mejor modelo.
        X_train (pd.DataFrame): Variables de entrenamiento sin escalar.
        X_train_scaled (pd.DataFrame): Variables de entrenamiento escaladas.
        y_train_scaled (np.ndarray): Variable objetivo escalada.
        x_scaler (StandardScaler): Escalador de variables X.
        y_scaler (MinMaxScaler): Escalador de la variable y.
        encoding (pd.Series): Target encoding aplicado.
        global_mean (float): Media global del precio.
        features (list): Variables utilizadas por el modelo.
        numerical_columns (list): Variables numéricas.
        categorical_columns (list): Variables categóricas.

    Returns:
        dict: Bundle con el modelo residual entrenado y su historial.
    """
    # Definición del bloque residual
    def residual_block(x, units, dropout_rate):
        shortcut = x

        # Primera capa
        x = Dense(units, activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout_rate)(x)

        # Segunda capa
        x = Dense(units, activation=None)(x)
        x = BatchNormalization()(x)

        # Ajuste del atajo si cambia la dimensión
        if shortcut.shape[-1] != units:
            shortcut = Dense(units, activation=None)(shortcut)

        # Conexión residual + activación
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x

    def create_mlp_residual_model(input_dim, dropout_rate):
        inputs = Input(shape=(input_dim,))
        x = Dense(128, activation=None)(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout_rate)(x)

        # Bloques residuales
        x = residual_block(x, 64, dropout_rate)
        x = residual_block(x, 32, dropout_rate)

        x = Dense(16, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        outputs = Dense(1, activation="linear")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer="nadam",
            loss="mae",
            metrics=["mae", "mse"]
        )
        return model

    model_mlp_residual = create_mlp_residual_model(input_dim=input_dim, dropout_rate=0.1)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, min_lr=1e-4, patience=20),
        ModelCheckpoint(
            filepath=path_file_mlp_residual,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    start_time = time.time()

    history_residual = model_mlp_residual.fit(
        X_train_scaled, y_train_scaled,
        epochs=500, batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    end_time = time.time()

    bundle = {
        "model_type": "keras",
        "model": None,
        "model_path": path_file_mlp_residual,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "encoding": encoding,
        "global_mean": global_mean,
        "features": features,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "training_time": (end_time - start_time),
        "training_medians": X_train[numerical_columns].median().to_dict(),
        "history": history_residual.history
    }

    return bundle

def train_ensamble_models(
    X_train, y_train, global_mean,
    maps_encoding, features, numerical_columns, log_transformed
):
    """
    Entrena un modelo de ensamble basado en stacking.

    Combina Random Forest y XGBoost como modelos base, y utiliza
    un modelo Ridge como meta-modelo entrenado sobre predicciones
    fuera de muestra (cross-validation).

    Args:
        X_train (pd.DataFrame): Variables de entrenamiento.
        y_train (pd.Series): Variable objetivo.
        global_mean (float): Media global del precio.
        maps_encoding (dict): Mapas de target encoding.
        features (list): Variables utilizadas por el modelo.
        numerical_columns (list): Variables numéricas.
        log_transformed (bool): Indica si se aplica log(1 + y).

    Returns:
        dict: Bundle con el ensamble entrenado y sus componentes.
    """
    # Ensamble modelos (Random Forest + XGBoost) 

    # Random Forest
    start_time = time.time()
    param_grid_rf = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    rf = RandomForestRegressor(random_state=123)

    grid_search_rf = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_rf,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=16,
        verbose=1
    )

    grid_search_rf.fit(X_train, y_train)

    # XGBoost
    param_grid_xgb = {
        "n_estimators": [100, 200],
        "max_depth": [2, 10],
        "learning_rate": [0.01, 0.03],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "gamma": [0, 1, 10],
        "reg_alpha": [0, 0.5],
        "reg_lambda": [1, 10],
        "tree_method": ["hist"]
    }

    xgb = XGBRegressor(random_state=123)

    grid_search_xgb = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid_xgb,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=16,
        verbose=1
    )

    grid_search_xgb.fit(X_train, y_train)


    # Extraer los parámetros del mejor modelo (tanto para RF como para XGBoost)
    rf_params = grid_search_rf.best_params_
    xgb_params = grid_search_xgb.best_params_

    # Creación de modelos (RF y XGB) usando los parámetros
    rf_clean = RandomForestRegressor(**rf_params, random_state=123)
    xgb_clean = XGBRegressor(**xgb_params, random_state=123)

    # Obtener predicciones usando cross_val_predict para que el meta-modelo no se memorice el entrenamiento
    preds_rf_train = cross_val_predict(rf_clean, X_train, y_train, cv=5)
    preds_xgb_train = cross_val_predict(xgb_clean, X_train, y_train, cv=5)

    # Crear el dataset de entrenamiento para el Meta-Modelo (usando las predicciones de los modelos base) 
    X_meta_train = np.column_stack([preds_rf_train, preds_xgb_train])

    # Entrenar los modelos base con todo el dataset de entrenamiento
    rf_clean.fit(X_train, y_train)
    xgb_clean.fit(X_train, y_train)

    # Entrenar el Meta-Modelo 
    meta_model = RidgeCV()
    meta_model.fit(X_meta_train, y_train)
    end_time = time.time()

    bundle = {
        "model_type": "ensamble",
        "model": None, 
        "model_path": None,
        
        # Los 3 motores del ensamble
        "base_model_rf": rf_clean,
        "base_model_xgb": xgb_clean,
        "meta_model": meta_model,
        
        # Los assets de preprocesamiento 
        "global_mean": global_mean,
        "maps_encoding": maps_encoding,
        "features": features,
        "meta_features": ["rf_pred", "xgb_pred"],
        "training_time": (end_time - start_time),
        "training_medians": X_train[numerical_columns].median().to_dict(),
        "log_transformed": log_transformed
    }

    return bundle














