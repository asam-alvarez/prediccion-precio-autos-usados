import os
import sys
from pathlib import Path
import pandas as pd
import joblib

# Configuración del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent

from functions.prediction import predict_raw_version_b

# Ruta del modelo
PATH_MODEL = PROJECT_ROOT / "models/xgboost_log.pkl"

# Casos de prueba
def load_test_cases():
    return {
        "datos_impecables": pd.DataFrame([{
            "brand": "alfa-romeo",
            "model": "Alfa Romeo GTV",
            "color": "red",
            "registration_date": "10/1995",
            "year": 1995,
            "power_kw": 148,
            "power_ps": 201,
            "transmission_type": "Manual",
            "fuel_type": "Petrol",
            "fuel_consumption_l_100km": "10.9 l/100 km",
            "mileage_in_km": 160500.0
        }]),
        "errores_formato": pd.DataFrame([{
            "brand": "alfa-romeo ",
            "model": "Alfa Romeo GTV",
            "color": "RED",
            "registration_date": "1995-10",
            "year": "1995",
            "power_kw": "148,0",
            "transmission_type": "manual",
            "fuel_type": "Petrol",
            "fuel_consumption_l_100km": "10,9 l/100 km",
            "mileage_in_km": "160500"
        }]),
        "columnas_faltantes": pd.DataFrame([{
            "brand": "alfa-romeo",
            "model": "Alfa Romeo GTV",
            "year": 1995,
            "power_kw": 148
        }]),
        "datos_minimos": pd.DataFrame([{
            "brand": "alfa-romeo",
            "model": "Alfa Romeo GTV"
        }])
    }

# Main
def main():
    if not PATH_MODEL.exists():
        raise FileNotFoundError("No se encontró el modelo XGBoost Log")

    bundle = joblib.load(PATH_MODEL)
    model = bundle["model"]

    test_cases = load_test_cases()
    results = []

    for name, df in test_cases.items():
        price = predict_raw_version_b(df, bundle, model)[0]
        results.append({
            "caso": name,
            "predicciones (€)": round(price, 2)
        })
    
    df_results = pd.DataFrame(results)

    print("\nPredicciones con XGBoost Log\n" + "-" * 40)
    print(df_results)


if __name__ == "__main__":
    main()
