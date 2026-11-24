import json
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

MODEL_VERSION = "1.0.0"

def main():
    # 1. Memuat dataset California Housing
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target  # Dalam satuan USD 100.000

    # 2. Split data sebelum preprocessing (anti-data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Baseline sederhana (DummyRegressor dengan median)
    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    mae_base = mean_absolute_error(y_test, y_pred_base)

    # 4. Pipeline Preprocessing + Model Regresi Linear
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    # 5. Evaluasi pada data test
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"=== Evaluasi Model (Versi {MODEL_VERSION}) ===")
    print(f"Baseline MAE (Median) : {mae_base:.4f} (USD {mae_base * 100_000:,.2f})")
    print(f"Model MAE             : {mae:.4f} (USD {mae * 100_000:,.2f})")
    print(f"Model RMSE            : {rmse:.4f} (USD {rmse * 100_000:,.2f})")
    print(f"Model R^2             : {r2:.4f}")

    # Contoh selisih aktual vs prediksi
    sample_idx = 0
    actual_val = y_test.iloc[sample_idx]
    pred_val = y_pred[sample_idx]
    print(f"\nContoh 1 Data Test:")
    print(f"Aktual   : {actual_val:.4f} (USD {actual_val * 100_000:,.2f})")
    print(f"Prediksi : {pred_val:.4f} (USD {pred_val * 100_000:,.2f})")
    print(f"Selisih  : {abs(actual_val - pred_val):.4f} (USD {abs(actual_val - pred_val) * 100_000:,.2f})")

    # 6. Simpan artefak model & metadata
    artefact = {
        "pipeline": pipeline,
        "version": MODEL_VERSION,
        "feature_names": list(X.columns),
        "metrics": {
            "baseline_mae": float(mae_base),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
    }
    joblib.dump(artefact, "model_california.joblib")
    print("\nArtefak berhasil disimpan ke 'model_california.joblib'.")

if __name__ == "__main__":
    main()
