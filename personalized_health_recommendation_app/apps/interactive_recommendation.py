import pandas as pd
import streamlit as st

# Load and preprocess the dataset
@st.cache_data
def load_data():
    # Load the dataset
    data = pd.read_csv('data/U.S._Chronic_Disease_Indicators_data.csv')
    
    # Filter for alcohol-related data
    alcohol_data = data[data['Topic'] == 'Alcohol'].copy()
    
    # Convert DataValue to numeric, handling any non-numeric values
    alcohol_data['DataValue'] = pd.to_numeric(alcohol_data['DataValue'], errors='coerce')
    
    # Ensure YearStart is an integer and handle missing or invalid values
    alcohol_data['YearStart'] = pd.to_numeric(alcohol_data['YearStart'], errors='coerce').fillna(0).astype(int)
    
    # Remove rows where DataValue is NaN or invalid, and exclude "United States"
    alcohol_data = alcohol_data.dropna(subset=['DataValue'])
    alcohol_data = alcohol_data[alcohol_data['LocationDesc'] != 'United States']
    
    return alcohol_data

# Define high-risk threshold
HIGH_RISK_THRESHOLD = 30.0

# Load the data
data = load_data()

# Streamlit App
def main():
    st.title("Youth Alcohol Use Recommendation System")
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    
    # Main content area
    page = st.sidebar.radio("Select a View", 
        ["Individual Recommendation", "High-Risk States Overview"])
    
    if page == "Individual Recommendation":
        individual_recommendation_page(data)
    else:
        high_risk_states_page(data)

def individual_recommendation_page(data):
    st.header("Individual Location Recommendation")
    
    # User inputs
    location = st.selectbox("Select a Location", 
        sorted(data['LocationDesc'].unique()))
    year = st.selectbox("Select a Year", 
        sorted(data['YearStart'].unique()))
    
    # Find the specific data point
    specific_data = data[(data['LocationDesc'] == location) & 
                         (data['YearStart'] == year)]
    
    if not specific_data.empty:
        data_value = specific_data['DataValue'].values[0]
        
        # Recommendation logic
        if data_value > HIGH_RISK_THRESHOLD:
            st.error(f"High Risk Alert for {location} in {year}!")
            st.warning(f"Alcohol use among youth is {data_value:.1f}%, which is above the recommended threshold.")
        else:
            st.success(f"Low Risk for {location} in {year}")
            st.info(f"Alcohol use among youth is {data_value:.1f}%")
    else:
        st.warning(f"No data available for {location} in {year}")

def high_risk_states_page(data):
    st.header("High-Risk States Overview")
    
    # Filter high-risk states
    high_risk_states = data[data['DataValue'] > HIGH_RISK_THRESHOLD]
    
    # Sort by DataValue in descending order
    high_risk_states = high_risk_states.sort_values('DataValue', ascending=False)
    
    # Display high-risk states
    if not high_risk_states.empty:
        st.dataframe(high_risk_states[['YearStart', 'LocationDesc', 'DataValue']])
        
        # Additional insights
        st.subheader("Key Insights")
        
        total_high_risk_states = len(high_risk_states['LocationDesc'].unique())
        
        highest_alcohol_use_row = high_risk_states.iloc[0]
        highest_alcohol_use_state = highest_alcohol_use_row['LocationDesc']
        highest_alcohol_use_year = highest_alcohol_use_row['YearStart']
        highest_alcohol_use_value = highest_alcohol_use_row['DataValue']
        
        st.write(f"Total High-Risk States: {total_high_risk_states}")
        st.write(f"Highest Alcohol Use: {highest_alcohol_use_state} in {highest_alcohol_use_year} at {highest_alcohol_use_value:.1f}%")
    else:
        st.info("No high-risk states found.")

# Run the app
if __name__ == "__main__":
    main()
