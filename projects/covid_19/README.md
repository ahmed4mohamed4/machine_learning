<div align="center">
    <h1> Developed by </h1>
    <div style="font-size: 24px; color: red;">
        <h2> Moaz Mohamed El Sayed </h2>
        <h2> Mahmoud Khaled Mahmoud </h2>
        <h2> Ahmed Mohamed Sayedahmed </h2>
    </div>
</div>

<br>
 
# COVID-19 Mortality Prediction

Machine learning project that predicts whether a patient in the COVID-19 dataset died.

> **For learning only.** This project is intended for educational use and experimentation, not for real-world clinical decision-making or deployment.

![COVID-19 illustration](images/covid.webp)

## Dataset

The data is the [COVID-19 Dataset on Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data).
The notebook creates a binary target named `DIED`:


## Project structure

```text
.
├── data/
│   └── covid.csv
├── images/
│   ├── covid.webp
│   ├── earth.webp
├── model/
└── README.md
└── requirements.txt
```

## Setup & Run locally

```bash
git clone https://github.com/ahmed4mohamed4/covid_19.git
```

```bash
cd covid_19
```

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt 
```
