import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="USD to IDR Exchange Rate Prediction",
    page_icon="📈",
    layout="wide"
)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/idr2usd-exchange-rate-processed.csv"
)

model = load_model()
df = load_data()

# ======================================================
# PREPARE DATA
# ======================================================
# Make datetime from Jan 2001
df["Date"] = pd.date_range(
    start="2001-01-01",
    periods=len(df),
    freq="MS"
)

df["Date"] = df["Date"][::-1].reset_index(drop=True)

X = df[["Index"]]
y = df["USD"]

# Model prediction for every historical data
df["Prediction"] = model.predict(X)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📈 Prediction")

# Month list until 2035
date_list = pd.date_range(
    start="2001-01-01",
    end="2035-12-01",
    freq="MS"
)

selected_date = st.sidebar.selectbox(
    "Select Prediction Month",
    date_list,
    index=len(df)
)

# Conversion datetime to index
selected_index = (
    (selected_date.year - 2001) * 12
    + selected_date.month
)

if st.sidebar.button("Predict"):

    prediction = model.predict(
        pd.DataFrame({"Index": [selected_index]})
    )[0]

    st.sidebar.success(f"""
### Prediction Result

📅 {selected_date.strftime('%B %Y')}

💵 **Rp {prediction:,.2f}**
""")

else:
    prediction = None

# ======================================================
# MAIN PAGE
# ======================================================
st.title("USD to IDR Exchange Rate Prediction")

st.caption(
    "Historical monthly exchange rate and machine learning prediction."
)

fig = go.Figure()

# ------------------------------------------------------
# Scatter Historis
# ------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["USD"],

        mode="markers",
        name="Historical Data",
        marker=dict(
            color="royalblue",
            size=7
        ),

        hovertemplate=
        "<b>%{x|%B %Y}</b><br>" +
        "Historical : %{y:,.0f} IDR<extra></extra>"
    )

)

# ------------------------------------------------------
# Prediction Line
# ------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Prediction"],

        mode="lines",
        name="Prediction",
        line=dict(
            color="crimson",
            width=3
        ),

        hovertemplate=
        "<b>%{x|%B %Y}</b><br>" +
        "Prediction : %{y:,.0f} IDR<extra></extra>"
    )

)

# ------------------------------------------------------
# Prediction Point
# ------------------------------------------------------
if prediction is not None:
    fig.add_trace(
        go.Scatter(
            x=[selected_date],
            y=[prediction],

            mode="markers",
            name="Selected Prediction",
            marker=dict(
                color="gold",
                size=16,
                symbol="star"
            ),

            hovertemplate=
            "<b>%{x|%B %Y}</b><br>" +
            "Prediction : %{y:,.0f} IDR<extra></extra>"
        )

    )

# ------------------------------------------------------
# Layout
# ------------------------------------------------------
fig.update_layout(
    template="plotly_white",
    height=700,
    hovermode="x unified",
    title="Historical Data vs Model Prediction",
    xaxis_title="Month",
    yaxis_title="Exchange Rate (IDR per USD)",
    legend=dict(
        orientation="h",
        y=1.05,
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)