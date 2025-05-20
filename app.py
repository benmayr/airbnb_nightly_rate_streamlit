import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from src.model import make_features

# Page config
st.set_page_config(
    page_title="NYC Airbnb Price Forecaster",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Center title
st.markdown("<h1 style='text-align: center;'>NYC Airbnb Price Forecaster</h1>", unsafe_allow_html=True)

# Cache data loading
@st.cache_resource
def load_data():
    try:
        df = pd.read_parquet("data/clean/listings.parquet")
        st.write("Data loaded successfully. Columns:", df.columns.tolist())
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/rf_price.pkl")
        st.write("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load data and model
df = load_data()
model = load_model()

if df is None or model is None:
    st.error("Failed to load data or model. Please check the files exist and are in the correct format.")
    st.stop()

# Sidebar
st.sidebar.header("Search Parameters")

# Get unique values for selectboxes
neighbourhood_groups = sorted(df['neighbourhood_group'].unique())
room_types = sorted(df['room_type'].unique())

# Sidebar inputs
selected_neighbourhood = st.sidebar.selectbox(
    "Neighbourhood Group",
    neighbourhood_groups
)

selected_room_type = st.sidebar.selectbox(
    "Room Type",
    room_types
)

selected_min_nights = st.sidebar.slider(
    "Minimum Nights",
    min_value=1,
    max_value=30,
    value=1
)

# Create input data for prediction
input_data = pd.DataFrame({
    'neighbourhood_group': [selected_neighbourhood],
    'room_type': [selected_room_type],
    'minimum_nights': [selected_min_nights],
    'availability_365': [365],  # Default to fully available
    'latitude': [df['latitude'].mean()],  # Default to mean location
    'longitude': [df['longitude'].mean()]
})

try:
    # Make features and predict
    X, _ = make_features(input_data)
    predicted_log_price = model.predict(X)[0]
    predicted_price = np.expm1(predicted_log_price)

    # Display predicted price
    st.metric(
        label="Predicted Nightly Price",
        value=f"${predicted_price:.2f}"
    )

    # Create map
    fig_map = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        color='price',
        size='availability_365',
        hover_name='neighbourhood',
        hover_data=['price', 'room_type', 'minimum_nights'],
        color_continuous_scale='Viridis',
        zoom=10,
        mapbox_style='carto-positron',
        labels={'price': 'Price ($)'}
    )

    # Center map on selected neighbourhood
    neighbourhood_center = df[df['neighbourhood_group'] == selected_neighbourhood][['latitude', 'longitude']].mean()
    fig_map.update_layout(
        mapbox=dict(
            center=dict(
                lat=neighbourhood_center['latitude'],
                lon=neighbourhood_center['longitude']
            ),
            zoom=10
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600,
        coloraxis_colorbar=dict(
            title="Price ($)",
            tickprefix="$",
            ticksuffix=""
        )
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # Create forecast
    seasonality = [1.00, 1.02, 1.03, 1.01, 0.98, 0.97, 1.00]
    forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=7, freq='D')
    forecast_prices = [predicted_price * factor for factor in seasonality]

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode='lines+markers',
        name='7-Day Forecast'
    ))

    fig_forecast.update_layout(
        title='7-Day Price Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Price ($)',
        showlegend=True
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Debug information:")
    st.write("Input data columns:", input_data.columns.tolist())
    st.write("DataFrame columns:", df.columns.tolist())
    st.write("Features shape:", X.shape if 'X' in locals() else "X not created")
    st.write("Model type:", type(model).__name__ if model else "No model") 