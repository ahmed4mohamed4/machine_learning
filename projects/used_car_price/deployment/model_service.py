"""Model loading, artifact inspection, and raw-input prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = APP_DIRECTORY.parent / "model" / "model.joblib"

ORDINAL_OPTIONS = {
    "condition": ["salvage", "fair", "good", "excellent", "like new", "new"],
    "size": ["sub-compact", "compact", "mid-size", "full-size"],
}


class ModelArtifactError(RuntimeError):
    """Raised when the saved artifact is not a usable self-contained pipeline."""


@dataclass(frozen=True)
class ModelMetadata:
    feature_names: tuple[str, ...]
    estimator_name: str
    onehot_options: dict[str, list[str]]


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the fitted pipeline once per process (the Streamlit caller caches it)."""
    if not model_path.is_file():
        raise ModelArtifactError(f"Model artifact was not found at {model_path}.")

    model = joblib.load(model_path)
    if not isinstance(model, Pipeline) or len(model.steps) < 2:
        raise ModelArtifactError("model.joblib must be a fitted scikit-learn Pipeline.")
    if "preprocessor" not in model.named_steps:
        raise ModelArtifactError("The pipeline does not contain its fitted 'preprocessor' step.")
    if not callable(getattr(model, "predict", None)):
        raise ModelArtifactError("The loaded artifact does not provide predict().")
    return model


def inspect_model(model: Pipeline) -> ModelMetadata:
    """Read fitted metadata from the artifact; never recreate preprocessing here."""
    feature_names = tuple(str(name) for name in getattr(model, "feature_names_in_", ()))
    if not feature_names:
        preprocessor = model.named_steps["preprocessor"]
        feature_names = tuple(str(name) for name in getattr(preprocessor, "feature_names_in_", ()))
    if not feature_names:
        raise ModelArtifactError("The fitted pipeline does not expose expected input feature names.")

    preprocessor = model.named_steps["preprocessor"]
    onehot_options: dict[str, list[str]] = {}
    for name, transformer, columns in getattr(preprocessor, "transformers_", ()):
        if name != "onehot" or transformer == "drop":
            continue
        encoder = transformer.named_steps.get("encoder")
        categories = getattr(encoder, "categories_", ())
        for column, values in zip(columns, categories):
            onehot_options[str(column)] = sorted(str(value) for value in values)

    return ModelMetadata(
        feature_names=feature_names,
        estimator_name=type(model.steps[-1][1]).__name__,
        onehot_options=onehot_options,
    )


def make_input_frame(values: dict[str, Any], metadata: ModelMetadata) -> pd.DataFrame:
    """Create the one-row DataFrame in the artifact's exact expected order."""
    missing = [name for name in metadata.feature_names if name not in values]
    if missing:
        raise ValueError(f"Missing required values for: {', '.join(missing)}.")
    return pd.DataFrame([{name: values[name] for name in metadata.feature_names}], columns=metadata.feature_names)


def predict_price(model: Pipeline, values: dict[str, Any], metadata: ModelMetadata) -> float:
    """Send raw, human-readable values directly through the fitted pipeline."""
    prediction = model.predict(make_input_frame(values, metadata))
    if len(prediction) != 1:
        raise ModelArtifactError("Expected a single prediction from the saved model.")
    return float(prediction[0])
