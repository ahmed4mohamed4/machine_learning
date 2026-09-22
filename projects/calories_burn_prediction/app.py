"""Streamlit portfolio application for exercise calorie prediction.

The estimator is loaded only for inference.  Training-time preprocessing is
reconstructed from the project dataset and notebook; the estimator is never
fit in this application.
"""

from pathlib import Path
import warnings

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "calories_Dataset.xlsx"
MODEL_PATH = ROOT / "model" / "model.joblib"

INPUT_COLUMNS = [
    "Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"
]
NUMERIC_MODEL_COLUMNS = ["Age", "Duration", "Heart_Rate", "Body_Temp", "BMI"]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the original project dataset."""
    return pd.read_excel(DATA_PATH)


@st.cache_resource(show_spinner="Loading the saved model...")
def load_model():
    """Load the persisted final estimator; do not train a model here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)


def clean_before_split(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's documented pre-split cleaning rules."""
    cleaned = data.drop(columns=["User_ID"]).drop_duplicates().copy()
    limits = {"Age": 100, "Height": 219, "Weight": 119, "Heart_Rate": 220, "Body_Temp": 43}
    for column, maximum in limits.items():
        cleaned = cleaned[cleaned[column].isna() | (cleaned[column] <= maximum)]
    return cleaned


@st.cache_resource(show_spinner=False)
def training_preprocessors():
    """Recreate fitted imputers/encoder from the original training partition.

    This mirrors the notebook's split (test_size=0.2, random_state=42), so a
    prediction never derives an imputation statistic from the user's row.
    """
    data = clean_before_split(load_data())
    features = data.drop(columns=["Calories"])
    target = data["Calories"]
    train_features, _, _, _ = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    numeric_imputer = SimpleImputer(strategy="median").fit(
        train_features[["Weight", "Heart_Rate"]]
    )
    categorical_imputer = SimpleImputer(strategy="most_frequent").fit(
        train_features[["Gender"]]
    )

    encoded_training_gender = pd.DataFrame(
        categorical_imputer.transform(train_features[["Gender"]]), columns=["Gender"]
    )
    encoder = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    ).fit(encoded_training_gender[["Gender"]])
    return numeric_imputer, categorical_imputer, encoder


def make_prediction_features(input_row: pd.DataFrame, model) -> pd.DataFrame:
    """Transform original user fields into the final model's unscaled input."""
    numeric_imputer, categorical_imputer, encoder = training_preprocessors()
    prepared = input_row.copy()
    prepared[["Weight", "Heart_Rate"]] = numeric_imputer.transform(
        prepared[["Weight", "Heart_Rate"]]
    )
    prepared[["Gender"]] = categorical_imputer.transform(prepared[["Gender"]])

    # BMI is engineered internally; users only supply height and weight.
    prepared["BMI"] = prepared["Weight"] / (prepared["Height"] / 100) ** 2
    numeric = prepared[NUMERIC_MODEL_COLUMNS].copy()
    encoded_gender = pd.DataFrame(
        encoder.transform(prepared[["Gender"]]),
        columns=encoder.get_feature_names_out(["Gender"]),
        index=prepared.index,
    )
    processed = pd.concat([numeric, encoded_gender], axis=1)

    expected_columns = list(model.feature_names_in_)
    processed = processed.reindex(columns=expected_columns)
    if processed.isna().any().any():
        raise ValueError("The processed prediction row contains missing values.")
    return processed


def model_results() -> pd.DataFrame:
    """Recorded train/test results from the project's model-comparison notebook."""
    rows = [
        ["Hist Gradient Boosting", 3.05, 3.53, -0.48, 5.03, 7.04, -2.01, 0.99, 0.99, 0.01],
        ["Extra Trees", 0.00, 3.80, -3.80, 0.00, 7.53, -7.52, 1.00, 0.99, 0.01],
        ["Random Forest", 1.43, 3.71, -2.29, 2.62, 7.57, -4.95, 1.00, 0.99, 0.01],
        ["Gradient Boosting", 4.34, 4.54, -0.20, 7.03, 8.06, -1.03, 0.99, 0.98, 0.00],
        ["KNN Regressor", 4.78, 5.98, -1.20, 7.68, 9.70, -2.01, 0.98, 0.98, 0.01],
        ["Decision Tree", 0.00, 5.39, -5.39, 0.00, 10.36, -10.36, 1.00, 0.97, 0.03],
        ["SVR (RBF)", 5.60, 5.76, -0.16, 12.45, 13.31, -0.86, 0.96, 0.96, 0.00],
        ["Ridge", 17.56, 18.05, -0.49, 25.82, 26.72, -0.91, 0.83, 0.82, 0.01],
        ["Linear Regression", 17.56, 18.05, -0.49, 25.82, 26.72, -0.91, 0.83, 0.82, 0.01],
        ["Lasso", 17.72, 18.18, -0.46, 25.89, 26.75, -0.86, 0.83, 0.82, 0.01],
        ["Elastic Net", 20.71, 21.09, -0.38, 28.06, 28.70, -0.64, 0.80, 0.79, 0.01],
        ["SVR (Linear)", 12.63, 12.97, -0.34, 33.46, 34.76, -1.30, 0.71, 0.70, 0.02],
    ]
    columns = ["Model", "MAE (Train)", "MAE (Test)", "MAE (Diff)", "RMSE (Train)", "RMSE (Test)", "RMSE (Diff)", "R² (Train)", "R² (Test)", "R² (Diff)"]
    return pd.DataFrame(rows, columns=columns)


def home_page():
    st.title("Exercise Calories Burn Prediction")
    st.subheader("Machine Learning Regression Portfolio Project")
    st.write(
        "This application estimates calories burned during exercise from user and "
        "exercise-related information. Its goal is to make the selected regression "
        "model easy to explore and use."
    )
    st.info("The model predicts the target variable: **Calories**.")
    st.write("**Final selected model:** HistGradientBoostingRegressor")
    a, b, c = st.columns(3)
    a.metric("Test MAE", "3.53")
    b.metric("Test RMSE", "7.04")
    c.metric("Test R²", "0.99")
    st.caption("Use the sidebar to review the data and workflow, or make a prediction.")


def about_page(data: pd.DataFrame):
    st.title("About the Data")
    st.write("The source dataset is loaded directly from the project repository.")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(data):,}")
    b.metric("Columns", len(data.columns))
    c.metric("Target", "Calories")
    st.subheader("Columns and data types")
    schema = pd.DataFrame({"Column": data.columns, "Data type": data.dtypes.astype(str).values})
    st.dataframe(schema, hide_index=True, use_container_width=True)
    st.subheader("Missing values")
    missing = data.isna().sum().rename("Missing values").reset_index(names="Column")
    st.dataframe(missing, hide_index=True, use_container_width=True)
    st.subheader("Basic statistical information")
    st.dataframe(data.describe(include="all").T, use_container_width=True)
    st.subheader("Feature roles")
    st.markdown("""
    - **Gender**: participant gender category.
    - **Age**: participant age.
    - **Height** and **Weight**: body measurements used to calculate BMI during preprocessing.
    - **Duration**: exercise duration.
    - **Heart_Rate**: recorded heart rate during exercise.
    - **Body_Temp**: recorded body temperature during exercise.
    - **Calories**: target variable—the calories burned during exercise.
    """)
    st.caption("User_ID is present in the source data but was removed before modeling because it is not useful for prediction.")


def eda_page(data: pd.DataFrame):
    st.title("Exploratory Data Analysis")
    analysis_data = clean_before_split(data)
    st.caption("Charts use the dataset after the documented pre-split cleaning rules; missing values are retained.")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(analysis_data, x="Calories", nbins=40, title="Calories distribution")
        fig.update_layout(yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        feature = st.selectbox("Numerical feature distribution", ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"], key="distribution")
        fig = px.histogram(analysis_data, x=feature, nbins=40, title=f"{feature} distribution")
        fig.update_layout(yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation heatmap")
    correlation_columns = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
    correlation = analysis_data[correlation_columns].corr(numeric_only=True)
    fig = go.Figure(go.Heatmap(
        z=correlation.values, x=correlation.columns, y=correlation.index,
        colorscale="RdBu", zmin=-1, zmax=1, text=correlation.round(2).values,
        texttemplate="%{text}", hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature relationship with Calories")
    relationship_feature = st.selectbox("Choose a feature", ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"], key="relationship")
    fig = px.scatter(analysis_data, x=relationship_feature, y="Calories", opacity=0.45,
                     title=f"{relationship_feature} vs Calories")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Calories by Gender")
    by_gender = analysis_data.dropna(subset=["Gender"]).groupby("Gender", as_index=False)["Calories"].mean()
    fig = px.bar(by_gender, x="Gender", y="Calories", text_auto=".2f", title="Average Calories by Gender")
    st.plotly_chart(fig, use_container_width=True)


def preprocessing_page():
    st.title("Data Preprocessing")
    st.subheader("Before the train/test split")
    st.markdown("""
    - Removed `User_ID`.
    - Removed duplicate rows.
    - Removed domain-based invalid values: `Age > 100`, `Height > 219`, `Weight > 119`, `Heart_Rate > 220`, and `Body_Temp > 43`.
    - Preserved missing values for later handling.
    """)
    st.subheader("After the train/test split")
    st.markdown("""
    - Imputed `Gender` with the most frequent value and `Weight` and `Heart_Rate` with their medians, fitted on the training split.
    - Created `BMI = Weight / (Height / 100) ** 2`.
    - Dropped `Height` and `Weight` after BMI was created.
    - One-hot encoded `Gender` with `drop="first"`, `handle_unknown="ignore"`, and `sparse_output=False`.
    - The final numerical features were `Age`, `Duration`, `Heart_Rate`, `Body_Temp`, and `BMI`.
    """)
    st.info("Scaling was tested for applicable models. The selected tree-based model uses the unscaled feature representation.")


def comparison_page():
    st.title("Model Comparison")
    st.write("The table below reproduces the recorded results from the project notebook.")
    results = model_results()
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.caption("Scaled models: Linear Regression, Ridge, Lasso, Elastic Net, SVR (Linear), SVR (RBF), and KNN Regressor. The remaining models used unscaled features.")
    st.warning("Models identified as overfitting: Extra Trees, Random Forest, and Decision Tree. SVR (Linear) was weaker, but its train/test results were relatively close.")
    st.subheader("Selected model: HistGradientBoostingRegressor")
    selected = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Train": [3.05, 5.03, 0.99],
        "Test": [3.53, 7.04, 0.99],
        "Difference": [-0.48, -2.01, 0.01],
    })
    st.dataframe(selected, hide_index=True, use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Test MAE", "3.53")
    b.metric("Test RMSE", "7.04")
    c.metric("Test R²", "0.99")


def prediction_page(model):
    st.title("Predict Calories Burned")
    st.write("Enter the original real-world features below. BMI is calculated automatically and is not an input field.")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("Gender", ["female", "male"])
            age = st.number_input("Age", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
            height = st.number_input("Height (cm)", min_value=1.0, max_value=219.0, value=175.0, step=0.1)
            weight = st.number_input("Weight (kg)", min_value=1.0, max_value=119.0, value=70.0, step=0.1)
        with right:
            duration = st.number_input("Duration", min_value=0.0, value=20.0, step=1.0)
            heart_rate = st.number_input("Heart Rate", min_value=1.0, max_value=220.0, value=100.0, step=1.0)
            body_temp = st.number_input("Body Temperature", min_value=1.0, max_value=43.0, value=40.0, step=0.1)
        submitted = st.form_submit_button("Estimate calories", type="primary", use_container_width=True)

    if submitted:
        original_input = pd.DataFrame([{
            "Gender": gender, "Age": age, "Height": height, "Weight": weight,
            "Duration": duration, "Heart_Rate": heart_rate, "Body_Temp": body_temp,
        }], columns=INPUT_COLUMNS)
        processed = make_prediction_features(original_input, model)
        prediction = float(model.predict(processed)[0])
        st.metric("Estimated Calories Burned", f"{prediction:.2f} kcal")
        with st.expander("Prediction input details"):
            st.write("BMI was calculated internally; Height and Weight were then removed before model prediction.")
            st.dataframe(processed, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="Exercise Calories Prediction", page_icon="🔥", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1180px; padding-top: 2.25rem; padding-bottom: 3rem;}
        [data-testid="stMetricValue"] {font-size: 1.7rem;}
    </style>""", unsafe_allow_html=True)
    if not MODEL_PATH.exists():
        st.error(f"Saved model not found: {MODEL_PATH}")
        st.stop()
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About the Data", "Exploratory Data Analysis", "Data Preprocessing", "Model Comparison", "Prediction"])
    st.sidebar.divider()
    st.sidebar.caption("Final model: HistGradientBoostingRegressor")

    model = load_model()
    if page == "Home":
        home_page()
    elif page == "About the Data":
        about_page(load_data())
    elif page == "Exploratory Data Analysis":
        eda_page(load_data())
    elif page == "Data Preprocessing":
        preprocessing_page()
    elif page == "Model Comparison":
        comparison_page()
    else:
        prediction_page(model)


if __name__ == "__main__":
    main()
