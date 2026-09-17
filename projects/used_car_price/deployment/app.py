"""Professional Streamlit interface for the saved used-car price pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from model_service import MODEL_PATH, ORDINAL_OPTIONS, ModelArtifactError, inspect_model, load_model, predict_price

APP_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = APP_DIRECTORY.parent / "data" / "vehicles.csv"
ANALYSIS_COLUMNS = (
    "price", "year", "manufacturer", "model", "condition", "cylinders", "fuel", "odometer",
    "title_status", "transmission", "drive", "size", "type", "paint_color", "region", "state",
)
NUMERIC_COLUMNS = ("price", "year", "odometer")

st.set_page_config(page_title="Used car price predictor", page_icon=":material/directions_car:", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=False)
def get_model_and_metadata():
    """Keep the fitted artifact in process memory after its first use."""
    pipeline = load_model()
    return pipeline, inspect_model(pipeline)


@st.cache_data(show_spinner="Loading local vehicle data…")
def load_dataset(dataset_path: str) -> tuple[pd.DataFrame, int]:
    """Load only analysis columns once; the raw CSV never leaves this machine."""
    path = Path(dataset_path)
    header = pd.read_csv(path, nrows=0)
    columns = [column for column in ANALYSIS_COLUMNS if column in header.columns]
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data, len(header.columns)


def dataset_or_message() -> tuple[pd.DataFrame | None, int | None]:
    if not DATASET_PATH.is_file():
        st.info("The saved prediction model is available, but the local source dataset is not present in this deployment.", icon=":material/database_off:")
        return None, None
    try:
        return load_dataset(str(DATASET_PATH))
    except Exception as exc:
        st.error(f"The local dataset could not be read: {type(exc).__name__}: {exc}")
        return None, None


def parse_numeric(value: str, label: str, *, non_negative: bool = False) -> tuple[float | None, str | None]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{label} must be numeric."
    if not math.isfinite(number):
        return None, f"{label} must be a finite number."
    if non_negative and number < 0:
        return None, f"{label} must be non-negative."
    return number, None


def categorical_input(label: str, feature: str, options: list[str]) -> str:
    return st.selectbox(label, options=[""] + options, format_func=lambda value: "Select a value" if value == "" else value, accept_new_options=True, key=f"prediction_{feature}")


def render_page_header(title: str, description: str, icon: str) -> None:
    st.title(title, icon=icon)
    st.caption(description)


def render_sidebar() -> str:
    with st.sidebar:
        st.title("AutoValue", icon=":material/directions_car:")
        st.caption("ML vehicle valuation workspace")
        st.space("small")
        page = st.radio("Navigation", ["Overview", "Price prediction", "Data understanding", "Dataset insights", "Data visualization", "Model information"], label_visibility="collapsed")
        st.space("medium")
        st.caption("Local-only analysis. No listing data is uploaded or shared.")
    return page


def render_overview() -> None:
    render_page_header("Used car price predictor", "Machine Learning-powered price estimation from real vehicle listings.", ":material/auto_graph:")
    with st.container(border=True):
        st.subheader("Price a vehicle with the fitted pipeline")
        st.write("Enter used-car listing details to receive the raw USD estimate from the saved machine-learning model. Explore the local dataset to understand the records behind the project.")
    data, column_count = dataset_or_message()
    if data is not None:
        prices = data.loc[data["price"].gt(0), "price"]
        with st.container(horizontal=True):
            st.metric("Dataset records", f"{len(data):,}", border=True)
            st.metric("Dataset columns", f"{column_count:,}", border=True)
            st.metric("Prediction target", "Listing price", border=True)
            st.metric("Median listed price", f"${prices.median():,.0f}" if not prices.empty else "Unavailable", border=True)
        st.caption("Dataset figures are calculated from the local `vehicles.csv` file.")
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Estimate a price", icon=":material/price_check:")
        st.write("Use raw listing values. Unknown categories are handled by the fitted pipeline.")
    with right.container(border=True):
        st.subheader("Explore the data", icon=":material/insights:")
        st.write("Review data quality, feature distributions, and price relationships from the local source data.")


def render_prediction_page() -> None:
    render_page_header("Price prediction", "Provide the same raw fields expected by the saved model pipeline.", ":material/price_check:")
    try:
        with st.spinner("Preparing the saved model…"):
            model, metadata = get_model_and_metadata()
    except Exception as exc:
        st.error("The saved model could not be loaded. Check the application dependencies and artifact.")
        st.caption(f"Model location: {MODEL_PATH}")
        st.exception(exc)
        return
    with st.form("prediction_form", border=False):
        vehicle, location = st.columns(2)
        with vehicle.container(border=True):
            st.subheader("Vehicle information", icon=":material/directions_car:")
            year = st.text_input("Year", placeholder="e.g. 2018", key="prediction_year")
            model_name = st.text_input("Model", placeholder="e.g. f-150", key="prediction_model")
            manufacturer = st.text_input("Manufacturer", placeholder="e.g. ford", key="prediction_manufacturer")
            odometer = st.text_input("Odometer (miles)", placeholder="e.g. 85000", key="prediction_odometer")
        with location.container(border=True):
            st.subheader("Location", icon=":material/location_on:")
            region = st.text_input("Region", placeholder="e.g. austin", key="prediction_region")
            state = st.text_input("State", placeholder="e.g. tx", key="prediction_state")
            st.caption("Use the listing's region and two-letter state value when available.")
        condition_col, specifications_col = st.columns(2)
        with condition_col.container(border=True):
            st.subheader("Vehicle condition", icon=":material/fact_check:")
            condition = categorical_input("Condition", "condition", ORDINAL_OPTIONS["condition"])
            size = categorical_input("Size", "size", ORDINAL_OPTIONS["size"])
            title_status = categorical_input("Title status", "title_status", metadata.onehot_options.get("title_status", []))
            paint_color = categorical_input("Paint color", "paint_color", metadata.onehot_options.get("paint_color", []))
        with specifications_col.container(border=True):
            st.subheader("Specifications", icon=":material/car_repair:")
            fields = (("Cylinders", "cylinders"), ("Fuel", "fuel"), ("Transmission", "transmission"), ("Drive", "drive"), ("Vehicle type", "type"))
            raw_values = {feature: categorical_input(label, feature, metadata.onehot_options.get(feature, [])) for label, feature in fields}
        submitted = st.form_submit_button("Estimate listing price", type="primary", icon=":material/auto_graph:", width="stretch")
    if not submitted:
        return
    year_value, year_error = parse_numeric(year, "Year")
    odometer_value, odometer_error = parse_numeric(odometer, "Odometer", non_negative=True)
    values: dict[str, object] = {"year": year_value, "odometer": odometer_value, "model": model_name.strip(), "region": region.strip(), "manufacturer": manufacturer.strip(), "state": state.strip(), "condition": condition, "size": size, "title_status": title_status, "paint_color": paint_color, **raw_values}
    errors = [error for error in (year_error, odometer_error) if error]
    errors.extend(f"{name.replace('_', ' ').capitalize()} is required." for name in metadata.feature_names if values.get(name) in (None, ""))
    if errors:
        st.error("Please correct the following before estimating:\n\n- " + "\n- ".join(errors), icon=":material/error:")
        return
    try:
        with st.spinner("Estimating listing price…"):
            price = predict_price(model, values, metadata)
    except (ValueError, ModelArtifactError) as exc:
        st.error(f"Prediction could not be completed: {exc}", icon=":material/error:")
        return
    except Exception as exc:
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}", icon=":material/error:")
        st.exception(exc)
        return
    result, summary = st.columns((1, 1.4))
    with result.container(border=True):
        st.subheader("Estimated listing price", icon=":material/payments:")
        st.metric("Model estimate", f"${price:,.0f}")
        st.caption("Raw USD prediction from the saved pipeline. It is an estimate, not an appraisal or guarantee.")
    with summary.container(border=True):
        st.subheader("Input summary", icon=":material/summarize:")
        st.write(f"**{values['year']:.0f} {values['manufacturer']} {values['model']}**")
        st.caption(f"{values['odometer']:,.0f} miles · {values['condition']} condition · {values['region']}, {values['state']} · {values['fuel']} · {values['transmission']}")


def render_data_understanding() -> None:
    render_page_header("Data understanding", "Data quality and structure from the local source file.", ":material/database:")
    data, column_count = dataset_or_message()
    if data is None:
        return
    numerical = [column for column in NUMERIC_COLUMNS if column in data]
    categorical = [column for column in data.columns if column not in numerical]
    with st.container(horizontal=True):
        st.metric("Rows", f"{len(data):,}", border=True)
        st.metric("Source columns", f"{column_count:,}", border=True)
        st.metric("Numerical fields", str(len(numerical)), border=True)
        st.metric("Categorical fields", str(len(categorical)), border=True)
        st.metric("Missing cells", f"{int(data.isna().sum().sum()):,}", border=True)
        st.metric("Duplicate analysis rows", f"{int(data.duplicated().sum()):,}", border=True)
    with st.expander("Missing-value overview", expanded=True, icon=":material/data_alert:"):
        missing = data.isna().sum().rename("Missing values").to_frame()
        missing["Missing rate"] = missing["Missing values"] / len(data) if len(data) else 0
        st.dataframe(missing.sort_values("Missing values", ascending=False), column_config={"Missing rate": st.column_config.NumberColumn(format="%.2f%%")})
        st.caption("Duplicate count is evaluated across locally loaded analytical fields, not text/image fields excluded for efficient analysis.")
    left, right = st.columns(2)
    with left:
        with st.expander("Feature data types", expanded=True, icon=":material/code:"):
            st.dataframe(pd.DataFrame({"Feature": data.columns, "Data type": data.dtypes.astype(str).values}), hide_index=True)
    with right:
        with st.expander("Target variable", expanded=True, icon=":material/paid:"):
            price = data.loc[data["price"].gt(0), "price"]
            st.table({"Target": "Listing price (`price`)", "Valid positive prices": f"{price.notna().sum():,}", "Missing prices": f"{data['price'].isna().sum():,}", "Median positive price": f"${price.median():,.0f}" if not price.empty else "Unavailable"})
    with st.expander("Basic descriptive statistics", icon=":material/query_stats:"):
        stats = data[numerical].describe().T
        st.dataframe(stats, column_config={column: st.column_config.NumberColumn(format="%.2f") for column in stats.columns})


def render_visualizations() -> None:
    render_page_header("Data visualization", "Interactive views of pricing patterns in the local vehicle listings.", ":material/monitoring:")
    data, _ = dataset_or_message()
    if data is None:
        return
    priced = data.loc[data["price"].gt(0)].copy()
    if priced.empty:
        st.warning("No positive price values are available for visualization.")
        return
    display_limit = priced["price"].quantile(0.99)
    display_prices = priced.loc[priced["price"].le(display_limit)]
    st.caption(f"Price charts exclude non-positive prices and visually cap values at the 99th percentile (${display_limit:,.0f}) to keep typical listings readable.")
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Listing price distribution")
        chart = px.histogram(display_prices, x="price", nbins=60, labels={"price": "Listing price (USD)"})
        chart.update_layout(bargap=0.03, height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(chart, width="stretch")
    with right.container(border=True):
        st.subheader("Median price by manufacturer")
        group = display_prices.dropna(subset=["manufacturer"]).groupby("manufacturer")["price"].agg(["median", "count"])
        group = group[group["count"] >= 25].sort_values("count", ascending=False).head(12).sort_values("median")
        chart = px.bar(group.reset_index(), x="median", y="manufacturer", orientation="h", labels={"median": "Median price (USD)", "manufacturer": "Manufacturer"})
        chart.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(chart, width="stretch")
    relation, composition = st.columns(2)
    with relation.container(border=True):
        st.subheader("Price and vehicle age")
        scatter = display_prices.dropna(subset=["year"])
        scatter = scatter.sample(25_000, random_state=42) if len(scatter) > 25_000 else scatter
        chart = px.scatter(scatter, x="year", y="price", opacity=0.35, labels={"year": "Model year", "price": "Listing price (USD)"})
        chart.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(chart, width="stretch")
    with composition.container(border=True):
        st.subheader("Vehicle type composition")
        counts = data["type"].dropna().value_counts().head(12).rename_axis("type").reset_index(name="Listings")
        chart = px.bar(counts, x="Listings", y="type", orientation="h", labels={"type": "Vehicle type"})
        chart.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(chart, width="stretch")
    with st.container(border=True):
        st.subheader("Price and odometer")
        scatter = display_prices.dropna(subset=["odometer"])
        scatter = scatter.sample(25_000, random_state=7) if len(scatter) > 25_000 else scatter
        chart = px.scatter(scatter, x="odometer", y="price", opacity=0.3, labels={"odometer": "Odometer (miles)", "price": "Listing price (USD)"})
        chart.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(chart, width="stretch")


def render_dataset_insights() -> None:
    render_page_header("Dataset insights", "A concise analytical view calculated from the local vehicle listings.", ":material/insights:")
    data, _ = dataset_or_message()
    if data is None:
        return
    prices = data.loc[data["price"].gt(0), "price"]
    mileage = data.loc[data["odometer"].ge(0), "odometer"]
    manufacturer = data["manufacturer"].dropna().mode()
    fuel = data["fuel"].dropna().mode()
    vehicle_type = data["type"].dropna().mode()
    with st.container(horizontal=True):
        st.metric("Median vehicle price", f"${prices.median():,.0f}" if not prices.empty else "Unavailable", border=True)
        st.metric("Median mileage", f"{mileage.median():,.0f} mi" if not mileage.empty else "Unavailable", border=True)
        st.metric("Most common manufacturer", manufacturer.iat[0] if not manufacturer.empty else "Unavailable", border=True)
        st.metric("Most common fuel", fuel.iat[0] if not fuel.empty else "Unavailable", border=True)
        st.metric("Most common vehicle type", vehicle_type.iat[0] if not vehicle_type.empty else "Unavailable", border=True)
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Leading manufacturers by listing count")
        manufacturers = data["manufacturer"].dropna().value_counts().head(10).rename_axis("Manufacturer").reset_index(name="Listings")
        st.dataframe(manufacturers, hide_index=True)
    with right.container(border=True):
        st.subheader("Fuel and transmission composition")
        fuel_counts = data["fuel"].dropna().value_counts().rename_axis("Fuel").reset_index(name="Listings")
        transmission_counts = data["transmission"].dropna().value_counts().rename_axis("Transmission").reset_index(name="Listings")
        st.dataframe(fuel_counts, hide_index=True)
        st.dataframe(transmission_counts, hide_index=True)
    st.caption("All figures are derived locally from the source CSV; no raw listings are sent to an external service.")


def render_model_information() -> None:
    render_page_header("Model information", "Verified details read from the fitted artifact and its preprocessing pipeline.", ":material/account_tree:")
    try:
        with st.spinner("Inspecting the saved model…"):
            model, metadata = get_model_and_metadata()
    except Exception as exc:
        st.error(f"The saved model could not be inspected: {type(exc).__name__}: {exc}")
        return
    preprocessor = model.named_steps["preprocessor"]
    with st.container(horizontal=True):
        st.metric("Estimator", metadata.estimator_name, border=True)
        st.metric("Input features", str(len(metadata.feature_names)), border=True)
        st.metric("Pipeline steps", str(len(model.steps)), border=True)
        st.metric("Artifact", MODEL_PATH.name, border=True)
    st.subheader("Pipeline structure", icon=":material/account_tree:")
    rows = []
    for name, transformer, columns in getattr(preprocessor, "transformers_", ()):
        if transformer == "drop":
            continue
        transformer_name = " → ".join(type(step).__name__ for _, step in transformer.steps) if hasattr(transformer, "steps") else type(transformer).__name__
        rows.append({"Stage": str(name), "Transformer": transformer_name, "Input columns": ", ".join(map(str, columns))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Raw input feature order", icon=":material/list:"):
        st.code("\n".join(metadata.feature_names), language="text")
    st.caption("All details above are inspected from the loaded artifact; no performance metrics are inferred or displayed.")


page = render_sidebar()
if page == "Overview":
    render_overview()
elif page == "Price prediction":
    render_prediction_page()
elif page == "Data understanding":
    render_data_understanding()
elif page == "Dataset insights":
    render_dataset_insights()
elif page == "Data visualization":
    render_visualizations()
else:
    render_model_information()
