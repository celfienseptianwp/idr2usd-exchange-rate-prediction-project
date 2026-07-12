# $\text{USD to IDR Exchange Rate Prediction}$

A machine learning web application for predicting the **USD to IDR exchange rate** using historical exchange rate data. The project compares multiple regression models, optimizes their hyperparameters using **Optuna**, selects the best-performing model, and deploys it through an interactive **Streamlit** application. Link for the application https://idr2usd-exchange-rate-prediction-project.streamlit.app/ 

## 📌 $\text{Project Overview}$

Forecasting exchange rates is an important task in economics and finance. This project aims to develop a machine learning model capable of predicting the USD to IDR exchange rate based on historical data.

The workflow includes:

1. Data preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Model use Multi-Layer Perceptron (MLP) Regressor
5. Hyperparameter optimization using Optuna
6. Model evaluation (R2 Score)
7. Model deployment with **Streamlit**

## 🛠️ $\text{Technologies}$

* Python
* Streamlit
* Scikit-learn
* Optuna
* Pandas
* NumPy
* Matplotlib
* Plotly
* Joblib

## 📂 Project Structure

```text
idr2usd-exchange-rate-prediction/
│
├── data/
│   └── processed.csv
│   └── raw.csv
│
├── models/
│   └── model.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   └── model_experiment.ipynb
│  
├── app.py               
├── README.md
└── requirements.txt
```