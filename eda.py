import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def show_eda(df):
    st.title("General Information About the Data")
    st.write("# Data Overview")
    st.write(df.head())

    st.write("### Descriptive Statistics")
    st.write(df.describe())

    st.write("### Distribution of Production")
    fig, ax = plt.subplots()
    df['Production (Tons)'].hist(ax=ax, bins=30)
    ax.set_title('Distribution of Production (Tons)')
    ax.set_xlabel('Production (Tons)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write("Distribution of Daily Prices")
    fig, ax = plt.subplots()
    sns.histplot(df['Daily Price'], bins=30, kde=True, ax=ax, color='skyblue')
    ax.set_title('Distribution of Daily Prices')
    ax.set_xlabel('Daily Price')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write("Distribution of Production (Tons)")
    fig, ax = plt.subplots()
    sns.histplot(df['Production (Tons)'], bins=30, kde=True, ax=ax, color='lightcoral')
    ax.set_title('Distribution of Production (Tons)')
    ax.set_xlabel('Production (Tons)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)



    st.title("City Wise Data")
    # Add a dropdown for city selection
    city_column = 'Location'  # Update this if the column name is different
    if city_column in df.columns:
        city = st.selectbox("Select a City", df[city_column].unique())

        # Filter data for the selected city
        city_data = df[df[city_column] == city]

        st.write(f"### Statistics for {city}")
        st.write(city_data.describe())

        st.write(f"### Distribution of Production in {city}")
        fig, ax = plt.subplots()
        city_data['Production (Tons)'].hist(ax=ax, bins=30)
        ax.set_title(f'Distribution of Production in {city}')
        ax.set_xlabel('Production (Tons)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)



        city_data = df[df['Location'] == city]

        st.write(f"### Distribution of Daily Prices in {city}")
        fig, ax = plt.subplots()
        sns.histplot(city_data['Daily Price'], bins=30, kde=True, ax=ax, color='lightblue')
        ax.set_title(f'Distribution of Daily Prices in {city}')
        ax.set_xlabel('Daily Price')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        st.write(f"### Distribution of Production (Tons) in {city}")
        fig, ax = plt.subplots()
        sns.histplot(city_data['Production (Tons)'], bins=30, kde=True, ax=ax, color='lightgreen')
        ax.set_title(f'Distribution of Production (Tons) in {city}')
        ax.set_xlabel('Production (Tons)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        city_data = df[df['Location'] == city]
        st.write(f"### Correlation Matrix for {city}")
        corr = city_data[['Daily Price', 'Production (Tons)', 'Rainfall (mm)', 'Max Temp', 'Humidity (%)']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title(f'Correlation Matrix for {city}')
        st.pyplot(fig)

    else:
        st.write(f"Column '{city_column}' not found in the dataset.")