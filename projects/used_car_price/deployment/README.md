# Used car price predictor deployment

This Streamlit app loads `../model/model.joblib` relative to this directory. It never rebuilds preprocessing or maps categories to numbers itself: one raw, human-readable row is assembled as a pandas `DataFrame` in the fitted pipeline's recorded feature order and is passed to `pipeline.predict()`.

## Requirements and compatibility

The artifact was serialized with **scikit-learn 1.6.1**, so use Python 3.10–3.13 and install the pinned deployment dependencies. Python 3.14 cannot run scikit-learn 1.6.1, and loading this artifact with scikit-learn 1.8.0 fails because a private scikit-learn serialization type was removed.

```bash
cd projects/used_car_price/deployment
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run locally

```bash
cd projects/used_car_price/deployment
source .venv/bin/activate
streamlit run app.py
```

The application uses a cached model resource, so the 5.5 GB artifact is loaded once per Streamlit process. Allow substantially more than 5.5 GB of RAM for the process and avoid multi-worker deployments unless the host has sufficient memory.

## Tests

```bash
cd projects/used_car_price/deployment
source .venv/bin/activate
PYTHONPATH=. python -m unittest discover -s tests -v
```

The smoke test loads the actual artifact and predicts from raw strings, including deliberately unseen category labels. The saved target encoder, ordinal encoder, and one-hot encoder provide the unknown-category behavior.

## Saved pipeline

Inspection of the artifact and training notebook shows a fitted scikit-learn `Pipeline` with a `preprocessor` `ColumnTransformer` followed by `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`. Its expected input columns are:

```text
year, odometer, model, region, manufacturer, state, condition, size,
cylinders, fuel, title_status, transmission, drive, type, paint_color
```

The app derives this ordered list from the loaded artifact at runtime rather than assuming it in prediction code. It also reads fitted one-hot encoder categories only to populate convenient UI choices; users can enter new raw categories, which the saved preprocessors handle safely.
