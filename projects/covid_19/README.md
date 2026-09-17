# COVID-19 Mortality Prediction

An educational classification project that predicts whether a patient record in the supplied COVID-19 dataset indicates death or survival. It includes a training notebook and a Streamlit interface named **CoviCare** for exploring a saved model prediction.

> This project is for learning and experimentation only. It is not a clinical decision-support system and must not be used for patient care.

![COVID-19 illustration](images/covid.webp)

## Data and target

The included [`data/covid_19.csv`](data/covid_19.csv) contains patient records and fields such as age, sex, pneumonia, diabetes, obesity, tobacco use, and ICU admission. The notebook derives the binary target from `DATE_DIED`:

- `9999-99-99` represents a recorded survivor.
- Any other value represents a recorded death.

Source: [COVID-19 Dataset on Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data).

## Training and analysis

[`model/main.ipynb`](model/main.ipynb) documents data preparation, exploratory analysis, class-imbalance handling with SMOTE/SMOTENC, and model training.

```bash
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
jupyter notebook model/main.ipynb
```

## Run the Streamlit app

The repository includes a trained artifact at `model/random_forest_model.joblib`. From this directory, install the app dependencies and start Streamlit:

```bash
python -m pip install streamlit pandas numpy scikit-learn joblib
streamlit run "streamlit app/app.py"
```

Use the local URL printed by Streamlit. The app collects patient attributes in the model's expected feature order and displays its prediction.

## Project structure

```text
covid_19/
├── data/
│   └── covid_19.csv
├── images/
│   ├── covid.webp
│   ├── earth.webp
├── model/
│   ├── main.ipynb
│   └── random_forest_model.joblib
├── streamlit app/
│   └── app.py
└── README.md
```
