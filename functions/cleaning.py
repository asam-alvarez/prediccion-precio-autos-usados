import pandas as pd
import os

def format_data(df, current_reference_year=2023, first_reference_year=1995):
    """
    Estandariza y formatea las columnas del dataset de autos usados.

    Realiza limpieza de texto, conversión de columnas numéricas,
    unificación de unidades de potencia (PS a kW), extracción del año
    de registro, cálculo de la antigüedad del vehículo y validación
    de variables categóricas.

    Args:
        df (pd.DataFrame): Dataset crudo de autos.
        current_reference_year (int, optional): Año de referencia para
            calcular la antigüedad del vehículo. Por defecto 2023.
        first_reference_year (int, optional): Año mínimo válido de fabricación.
            Por defecto 1995.

    Returns:
        pd.DataFrame: Dataset formateado y estandarizado.
    """

    # Formatear las columnas 
    if "brand" in df.columns:
        df["brand"] = df["brand"].astype(str).str.lower().str.strip()

    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.lower().str.strip()

    if "color" in df.columns:
        df["color"] = df["color"].astype(str).str.lower().str.strip()

    if "transmission_type" in df.columns:
        df["transmission_type"] = df["transmission_type"].astype(str).str.strip().str.capitalize()

    if "mileage_in_km" in df.columns:
        if df["mileage_in_km"].dtype == "object":
            df["mileage_in_km"] = df["mileage_in_km"].astype(str).str.lower().str.replace("km", "").str.strip()
            df["mileage_in_km"] = df["mileage_in_km"].str.replace(",", ".", regex=False)

        df["mileage_in_km"] = pd.to_numeric(df["mileage_in_km"], errors="coerce")

    if "power_kw" in df.columns:
        if df["power_kw"].dtype == "object":
            df["power_kw"] = df["power_kw"].astype(str).str.lower().str.replace("kw", "").str.strip()
            df["power_kw"] = df["power_kw"].str.replace(",", ".", regex=False)

        df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")

    if "power_kw" in df.columns and "power_ps" in df.columns:
        df["power_kw"] = df["power_kw"].fillna(df["power_ps"] * 0.735499)
    elif "power_kw" not in df.columns and "power_ps" in df.columns:
        df["power_kw"] = df["power_ps"] * 0.735499

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "year" in df.columns and "registration_date" in df.columns:
        extracted_year = df["registration_date"].astype(str).str.extract(r'(\d{4})')[0]
        df["year"] = df["year"].fillna(pd.to_numeric(extracted_year, errors="coerce"))

    if "year" in df.columns:
        df["year"] = (df["year"].where((df["year"] >= first_reference_year) & (df["year"] <= current_reference_year)))
        df["age"] = (current_reference_year - df["year"]).astype(float)
    
    if "fuel_consumption_l_100km" in df.columns:
        if df["fuel_consumption_l_100km"].dtype == "object":
            extracted = df["fuel_consumption_l_100km"].astype(str).str.extract(r"(\d+[.,]?\d*)")[0]
            extracted = extracted.str.replace(",", ".", regex=False)
            df["fuel_consumption_l_100km"] = pd.to_numeric(extracted, errors="coerce")

    if "price_in_euro" in df.columns:
        df["price_in_euro"] = pd.to_numeric(df["price_in_euro"], errors="coerce")
        df["price_in_euro"] = df["price_in_euro"].where(df["price_in_euro"] > 0)

    if "fuel_type" in df.columns:
        valid_fuel_types = [
            "Petrol", "Diesel", "Hybrid", "Diesel Hybrid",
            "Electric", "LPG", "CNG", "Hydrogen", "Ethanol", "Other"
        ]
        
        df["fuel_type"] = df["fuel_type"].astype(str).str.title()
        df["fuel_type"] = df["fuel_type"].apply(lambda x: x if x in valid_fuel_types else "Unknown")

    return df


def filter_data(df):
    """
    Filtra y elimina datos inválidos o redundantes del dataset.

    Elimina filas con valores nulos en variables críticas,
    descarta columnas irrelevantes o redundantes y
    elimina registros duplicados.

    Args:
        df (pd.DataFrame): Dataset previamente formateado.

    Returns:
        pd.DataFrame: Dataset filtrado y listo para modelado.
    """

    # Filtrar columnas que sufrirán modificaciones
    redundant_columns=[
        "Unnamed: 0", "registration_date", "power_ps", "offer_description", "fuel_consumption_g_km", "year"
    ]

    cols_to_drop_nulls = [
        "color", "price_in_euro", "power_kw", "mileage_in_km",
        "fuel_consumption_l_100km", "fuel_consumption_g_km"
    ]

    # Eliminar filas con valores nulos, columnas redundantes (o irrelevantes) y filas duplicadas
    df = df.dropna(subset=[c for c in cols_to_drop_nulls if c in df.columns])

    df = df.drop(columns=[c for c in redundant_columns if c in df.columns])
    
    df = df.drop_duplicates()

    return df


def clean_for_training(path_data, path_cleaned_data):
    """
    Carga, limpia y guarda el dataset para el entrenamiento de modelos.

    Lee el dataset crudo desde disco, aplica los pasos de
    formateo y filtrado, y guarda la versión limpia en un archivo CSV.

    Args:
        path_data (str): Ruta al archivo CSV con los datos crudos.
        path_cleaned_data (str): Ruta donde se guardará el dataset limpio.

    Returns:
        pd.DataFrame: Dataset limpio utilizado para el entrenamiento.

    Raises:
        FileNotFoundError: Si el archivo de datos crudos no existe.
    """

    # Verificación de la existencia del archivo
    if not os.path.exists(path_data):
        raise FileNotFoundError(f"No se encontró el archivo original en {path_data}")
    
    # Cargar el archivo
    df = pd.read_csv(path_data)

    # Formatear los datos
    df = format_data(df)

    # Eliminar columnas innecesarias
    df = filter_data(df)
    
    # Guardar en un csv los datos limpios
    os.makedirs(os.path.dirname(path_cleaned_data), exist_ok=True)
    df.to_csv(path_cleaned_data, index=False)

    return df
