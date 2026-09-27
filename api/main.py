from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lightgbm.joblib"

num_cols = [f"integer_feature_{i}" for i in range(1, 14)]
cat_cols = [f"categorical_feature_{i}" for i in range(1, 27)]

# load the model once when the API starts
saved = joblib.load(MODEL_PATH)
model = saved["model"]
counts = saved["counts"]

app = FastAPI(title="Criteo click prediction")


class Ad(BaseModel):
    # all fields are optional: missing values are normal in this data
    integer_feature_1: float | None = None
    integer_feature_2: float | None = None
    integer_feature_3: float | None = None
    integer_feature_4: float | None = None
    integer_feature_5: float | None = None
    integer_feature_6: float | None = None
    integer_feature_7: float | None = None
    integer_feature_8: float | None = None
    integer_feature_9: float | None = None
    integer_feature_10: float | None = None
    integer_feature_11: float | None = None
    integer_feature_12: float | None = None
    integer_feature_13: float | None = None
    categorical_feature_1: str | None = None
    categorical_feature_2: str | None = None
    categorical_feature_3: str | None = None
    categorical_feature_4: str | None = None
    categorical_feature_5: str | None = None
    categorical_feature_6: str | None = None
    categorical_feature_7: str | None = None
    categorical_feature_8: str | None = None
    categorical_feature_9: str | None = None
    categorical_feature_10: str | None = None
    categorical_feature_11: str | None = None
    categorical_feature_12: str | None = None
    categorical_feature_13: str | None = None
    categorical_feature_14: str | None = None
    categorical_feature_15: str | None = None
    categorical_feature_16: str | None = None
    categorical_feature_17: str | None = None
    categorical_feature_18: str | None = None
    categorical_feature_19: str | None = None
    categorical_feature_20: str | None = None
    categorical_feature_21: str | None = None
    categorical_feature_22: str | None = None
    categorical_feature_23: str | None = None
    categorical_feature_24: str | None = None
    categorical_feature_25: str | None = None
    categorical_feature_26: str | None = None


# same preprocessing as in the notebook (part 7)
def preprocess(df):
    X = pd.DataFrame(index=df.index)
    for col in num_cols:
        X[col] = np.log1p(df[col].astype(float).fillna(0).clip(lower=0))
    for col in cat_cols:
        X[col] = df[col].map(counts[col]).fillna(0)
    return X


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(ad: Ad):
    df = pd.DataFrame([ad.model_dump()])
    X = preprocess(df)
    probability = model.predict_proba(X)[0, 1]
    return {"click_probability": float(probability)}
