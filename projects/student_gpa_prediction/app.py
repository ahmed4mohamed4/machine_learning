"""Interactive Streamlit app for the Student GPA Prediction project."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "data.csv"
FEATURES = [
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "Extracurricular",
    "Sports",
    "Music",
    "ParentalSupport",
]
FEATURE_LABELS = {
    "StudyTimeWeekly": "Weekly study time (hours)",
    "Absences": "Absences this year",
    "Tutoring": "Tutoring",
    "Extracurricular": "Extracurricular activities",
    "Sports": "Sports participation",
    "Music": "Music participation",
    "ParentalSupport": "Parental support",
}
SUPPORT_LABELS = {
    0: "None",
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very high",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the bundled project dataset."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def train_model():
    """Fit the same leakage-free model selected in the project notebook."""
    data = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data["GPA"], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
    }
    return model, metrics


def grade_band(gpa: float) -> tuple[str, str]:
    if gpa >= 3.5:
        return "Excellent standing", "🌟"
    if gpa >= 2.5:
        return "Good standing", "✅"
    if gpa >= 1.5:
        return "Developing standing", "📈"
    return "Needs additional support", "🤝"


def input_frame(
    study_time: float,
    absences: int,
    tutoring: bool,
    extracurricular: bool,
    sports: bool,
    music: bool,
    parental_support: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [[study_time, absences, int(tutoring), int(extracurricular), int(sports), int(music), parental_support]],
        columns=FEATURES,
    )


st.markdown(
    """
    <style>
      .block-container { max-width: 1180px; padding-top: 2.5rem; }
      .hero { padding: 1.7rem 1.9rem; border-radius: 18px; background: linear-gradient(120deg, #12355b, #216c65); color: white; margin-bottom: 1.5rem; }
      .hero h1 { margin: 0 0 .3rem; font-size: 2.35rem; }
      .hero p { margin: 0; opacity: .93; font-size: 1.05rem; }
      .small-note { color: #667085; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

data = load_data()
model, metrics = train_model()

with st.sidebar:
    st.title("🎓 GPA Predictor")
    page = st.radio("Navigate", ["Predict GPA", "Explore the data", "About the model"])
    st.divider()
    st.caption("Built from the included Student Performance dataset.")

if page == "Predict GPA":
    st.markdown(
        """<section class="hero"><h1>Plan for academic success</h1>
        <p>Estimate GPA from study habits, attendance, support, and activities.</p></section>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Student profile")
        st.caption("Enter the student's current habits. Averages from the dataset are used as starting values.")
        study_time = st.number_input(
            "Weekly study time (hours)", min_value=0.0, max_value=100.0,
            value=float(round(data["StudyTimeWeekly"].median(), 1)), step=0.5,
        )
        absences = st.number_input(
            "Absences this year", min_value=0, max_value=100,
            value=int(data["Absences"].median()), step=1,
        )
        parental_support = st.select_slider(
            "Parental support", options=list(SUPPORT_LABELS), value=2,
            format_func=lambda value: SUPPORT_LABELS[value],
        )
        st.markdown("**Activities and support**")
        c1, c2 = st.columns(2)
        with c1:
            tutoring = st.toggle("Receives tutoring", value=False)
            sports = st.toggle("Participates in sports", value=False)
        with c2:
            extracurricular = st.toggle("Extracurricular activities", value=False)
            music = st.toggle("Participates in music", value=False)

    profile = input_frame(
        study_time, absences, tutoring, extracurricular, sports, music, parental_support
    )
    predicted_gpa = float(np.clip(model.predict(profile)[0], 0, 4))
    band, icon = grade_band(predicted_gpa)

    with right:
        st.subheader("Estimated outcome")
        st.metric("Predicted GPA", f"{predicted_gpa:.2f}", help="GPA is reported on a 0.00–4.00 scale.")
        st.info(f"{icon} **{band}**")
        percentile = float((data["GPA"] <= predicted_gpa).mean() * 100)
        st.metric("Dataset percentile", f"{percentile:.0f}th", help="Share of students in this dataset with a GPA at or below this estimate.")
        st.progress(int(percentile))
        st.caption("This is an educational estimate, not an assessment of a student's ability or potential.")

    st.divider()
    st.subheader("What-if comparison")
    comparison = profile.copy()
    comparison["StudyTimeWeekly"] = min(study_time + 2, 100)
    comparison["Absences"] = max(absences - 2, 0)
    improved_gpa = float(np.clip(model.predict(comparison)[0], 0, 4))
    change = improved_gpa - predicted_gpa
    st.write(
        f"With **2 additional weekly study hours** and **2 fewer absences**, the model estimates a GPA of "
        f"**{improved_gpa:.2f}** ({change:+.2f})."
    )

elif page == "Explore the data":
    st.markdown(
        """<section class="hero"><h1>Understand the dataset</h1>
        <p>Explore the 2,392 anonymized student records used by the predictor.</p></section>""",
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.metric("Students", f"{len(data):,}")
    b.metric("Average GPA", f"{data['GPA'].mean():.2f}")
    c.metric("Average absences", f"{data['Absences'].mean():.1f}")

    chart_col, table_col = st.columns([1.15, 1])
    with chart_col:
        st.subheader("Study time and GPA")
        st.scatter_chart(data, x="StudyTimeWeekly", y="GPA", color="ParentalSupport", height=390)
    with table_col:
        st.subheader("Feature relationships")
        relationships = pd.DataFrame({
            "Feature": [FEATURE_LABELS[col] for col in FEATURES],
            "Correlation with GPA": [data[col].corr(data["GPA"]) for col in FEATURES],
        }).set_index("Feature")
        st.bar_chart(relationships, horizontal=True, height=390)

    st.subheader("Dataset preview")
    st.dataframe(data.drop(columns=["GradeClass"]), use_container_width=True, height=280)

else:
    st.markdown(
        """<section class="hero"><h1>About this prediction</h1>
        <p>A transparent baseline model built for learning and exploration.</p></section>""",
        unsafe_allow_html=True,
    )
    st.subheader("Model performance")
    a, b, c = st.columns(3)
    a.metric("Holdout R²", f"{metrics['r2']:.3f}")
    b.metric("Mean absolute error", f"{metrics['mae']:.3f}")
    c.metric("RMSE", f"{metrics['rmse']:.3f}")
    st.caption("Metrics are calculated on a fixed 20% test split (random state 42). Lower error values are better.")

    st.subheader("How it works")
    st.write(
        "A linear regression model is trained from the bundled dataset whenever the app starts. "
        "It uses seven inputs selected in the project notebook: study time, absences, tutoring, extracurricular activities, sports, music, and parental support."
    )
    st.warning(
        "`GradeClass` is intentionally excluded because it is derived from GPA and would leak the answer into the model. "
        "Do not use this app for admissions, discipline, or other high-stakes decisions."
    )
    st.subheader("Feature influence")
    coefficients = pd.DataFrame({
        "Feature": [FEATURE_LABELS[col] for col in FEATURES],
        "Model coefficient": model.coef_,
    }).set_index("Feature").sort_values("Model coefficient")
    st.bar_chart(coefficients, horizontal=True)
    st.caption("Coefficients describe associations in this dataset; they do not establish causation.")
