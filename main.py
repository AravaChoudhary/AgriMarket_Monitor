import joblib
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import datetime  # Import datetime module
from eda import show_eda

# Load the fitted preprocessor
preprocessor = joblib.load('preprocessor.pkl')

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the model
input_size = 33  # Make sure this matches your model's input size
hidden_size = 64
output_size = 1
model = FeedForwardNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("agri_model.pth"))
model.eval()  # Set the model to evaluation mode

# Define the Streamlit UI
st.sidebar.title("Navigator")
option = st.sidebar.radio("Select a section", ["Model Prediction", "Data Analysis (EDA)"])

st.sidebar.write("""
## 📋 Instructions for Using the Model
This application predicts the daily price of agricultural commodities based on various input features. 
Please select a section from the navigation bar to use the application.

### 🧮 Model Prediction Section
- **🗓️ Date**: The date for which you want to predict the price.
- **🌾 Commodity**: The type of commodity (e.g., gram, rice) whose price you want to predict.
- **📍 Location**: The location where the commodity is being assessed.
- **📏 Unit**: The unit of measurement for production (e.g., kg, quintal).
- **📊 Production (Tons)**: The amount of production in tons.
- **🌧️ Rainfall (mm)**: Amount of rainfall, which can affect crop yields.
- **🌡️ Max Temp (°C)** and **🌡️ Min Temp (°C)**: Maximum and minimum temperatures during the growing period.
- **🌱 Sowing Date**: The date when the crop was sown.
- **🌾 Harvest Date**: The date when the crop was harvested.
- **💧 Humidity (%)**: The humidity level which can influence crop growth.
- **⚠️ Extreme Event**: Any extreme weather event (e.g., flood, drought) that occurred.
- **🎉 Festival/Holiday**: Local festivals or holidays that may affect commodity prices.
- **🔍 Commodity Relevance**: Indicates the importance of the commodity in the given location.
- **💬 Sentiment Score**: Public sentiment towards the commodity, which may influence market prices.
- **📅 Key Event**: Significant events (e.g., government policies) that could impact prices.
""")

if option == "Model Prediction":
    st.title("Agricultural Commodity Price Prediction")

    # Input fields in columns
    col1, col2 = st.columns(2)

    with col1:
        date = st.date_input("Date", datetime.date(2022, 4, 11))
        commodity = st.selectbox("Commodity", ["gram", "rice", "wheat", "corn"])
        location = st.selectbox("Location", ["Mumbai", "Delhi", "Chennai", "Bangalore"])
        unit = st.selectbox("Unit", ["kg", "quintal", "tonne"])
        production_tons = st.number_input("Production (Tons)", value=20580.85)
        rainfall = st.number_input("Rainfall (mm)", value=141.61)
        max_temp = st.number_input("Max Temp (°C)", value=26.23)
        min_temp = st.number_input("Min Temp (°C)", value=22.52)

    with col2:
        sowing_date = st.date_input("Sowing Date", datetime.date(2015, 5, 14))
        harvest_date = st.date_input("Harvest Date", datetime.date(2015, 9, 9))
        humidity = st.number_input("Humidity (%)", value=54.22)
        extreme_event = st.selectbox("Extreme Event", ["None", "Flood", "Drought", "Heatwave"])
        festival_holiday = st.selectbox("Festival/Holiday", ["Holi", "Diwali", "None"])
        commodity_relevance = st.selectbox("Commodity Relevance", ["gram", "rice", "wheat", "corn"])
        sentiment_score = st.selectbox("Sentiment Score", ["Positive", "Neutral", "Negative"])
        key_event = st.selectbox("Key Event", ["Government Policy", "Market Intervention", "None"])

    # Collect input data in a dictionary
    input_data = {
        'Date': str(date),
        'Commodity': commodity,
        'Location': location,
        'Unit': unit,
        'Sowing Date': str(sowing_date),
        'Harvest Date': str(harvest_date),
        'Production (Tons)': production_tons,
        'Rainfall (mm)': rainfall,
        'Max Temp': max_temp,
        'Min Temp': min_temp,
        'Humidity (%)': humidity,
        'Extreme Event': extreme_event,
        'Festival/Holiday': festival_holiday,
        'Commodity Relevance': commodity_relevance,
        'Sentiment Score': sentiment_score,
        'Key Event': key_event
    }

    # Define columns for date handling
    date_columns = ['Date', 'Sowing Date', 'Harvest Date']

    # Button to trigger prediction
    if st.button("Predict"):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Process the dates
        input_df['Date'] = pd.to_datetime(input_df['Date'], errors='coerce')
        input_df['Sowing Date'] = pd.to_datetime(input_df['Sowing Date'], errors='coerce')
        input_df['Harvest Date'] = pd.to_datetime(input_df['Harvest Date'], errors='coerce')

        # Drop date columns since they're not used directly in the model
        input_features = input_df.drop(columns=date_columns)

        # Preprocess input
        input_processed = preprocessor.transform(input_features)

        # Convert to tensor for prediction
        input_tensor = torch.tensor(input_processed, dtype=torch.float32)

        # Perform prediction
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)
            st.write(f"Predicted Daily Price: ₹{prediction.item():.2f}")

elif option == "Data Analysis (EDA)":
    st.title("Data Analysis (EDA)")

    # Load and display sample data (Replace with your dataset path)
    df = pd.read_csv('agriculture_data.csv')  # Replace with your actual data path

    show_eda(df)