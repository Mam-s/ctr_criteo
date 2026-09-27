# Criteo click prediction

Predict the probability that a user clicks on an ad, using the Criteo 1TB Click Logs dataset.
The best model (LightGBM) is served with a FastAPI app that runs in a Docker container.

## Data

Dataset: [criteo/CriteoClickLogs](https://huggingface.co/datasets/criteo/CriteoClickLogs) on Hugging Face (license CC BY-NC-SA 4.0).

Each row is an ad that was displayed, with a label (1 = clicked, 0 = not clicked), 13 integer features and 26 categorical features. The feature names and values are anonymised.

The full dataset is about 1 TB (24 days), so only one file per day is used:

| split | day | rows |
|---|---|---|
| train | 2015-02-15 and 2015-02-16 | 2,308,265 |
| validation | 2015-02-17 | 769,126 |
| test | 2015-02-18 | 706,671 |

The split is done by day, so the models are always evaluated on days they have never seen.

## Project structure

```
ctr_criteo/
├── scripts/download_data.py      download the data from Hugging Face
├── notebooks/exploration.ipynb   data exploration, models and evaluation
├── api/main.py                   FastAPI app
├── tests/test_api.py             tests for the API
├── models/                       saved models (created by the notebook, not in git)
├── requirements.txt              packages needed by the API
└── Dockerfile
```

## Results

Only 3.2% of the ads are clicked, so the models are evaluated with log loss (lower is better) and ROC AUC (higher is better). The baseline always predicts the average click rate of the train set.

| model | validation log loss | validation ROC AUC | test log loss | test ROC AUC |
|---|---|---|---|---|
| baseline | 0.1426 | 0.500 | 0.1406 | 0.500 |
| logistic regression | 0.1341 | 0.702 | | |
| LightGBM | **0.1284** | **0.750** | **0.1270** | **0.748** |
| transformer | 0.1299 | 0.738 | 0.1285 | 0.736 |

LightGBM is the best model and the fastest to train (about 30 seconds), so it is the one used by the API.

## How to run

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install huggingface_hub matplotlib seaborn torch ipykernel pytest httpx
```

### 2. Download the data

```bash
python scripts/download_data.py
```

This downloads 4 files (about 230 MB) into `data/raw/`.

### 3. Run the notebook

Run all the cells of `notebooks/exploration.ipynb`. The last part saves the models into `models/`:

- `models/lightgbm.joblib`: the LightGBM model and the value counts used to encode the categorical columns
- `models/transformer.pt`: the transformer weights

The API needs `models/lightgbm.joblib`, so the notebook has to be run before the next steps.

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

Then open http://localhost:8000/docs to test it in the browser.

### 5. Run the tests

```bash
pytest
```

### 6. Run with Docker

```bash
docker build -t ctr-api .
docker run -p 8000:8000 ctr-api
```

## API

`GET /health` returns `{"status": "ok"}`.

`POST /predict` takes an ad and returns the click probability. All the fields are optional because missing values are common in this data.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"integer_feature_1": 13, "integer_feature_2": 454, "categorical_feature_2": "070c56a5"}'
```

```json
{"click_probability": 0.1858604806353454}
```

## Possible improvements

- Use more data (more files per day)
- Tune the LightGBM hyperparameters
- Try other encodings for the categorical columns (target encoding, embeddings)
- Add a GitHub Actions workflow that runs the tests
