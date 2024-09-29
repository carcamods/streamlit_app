import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and scaler with proper error handling
model = None
scaler = None
df = None
try:
    model = pickle.load(open('random_forest_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))  
    df = pd.read_csv('processed_validation_df.csv')
except Exception as e:
    st.error(f"Error loading model, scaler, or validation data: {e}")
    st.stop()  # Stop further execution if loading fails

# Load the df that contains the raw values per summoner_id
try:
    raw_df = pd.read_csv('validation_df.csv')
    raw_df['game_creation_dt'] = pd.to_datetime(raw_df['game_creation_dt'])  # Convert to datetime if not already
except Exception as e:
    st.error(f"Error loading raw data: {e}")
    st.stop()  # Stop further execution if raw data loading fails


# Streamlit webpage title
st.title('Summoner Activity Prediction')

# Creating a dropdown for summoner IDs
summoner_id = st.selectbox('Select a Summoner ID:', df['summoner_id'].unique())

# List of features as expected by the model
feature_columns = list(scaler.feature_names_in_)

# When a summoner is selected, display the prediction and graphs
if st.button('Predict Activity'):
    # Extract the row corresponding to the selected summoner_id
    summoner_row = df[df['summoner_id'] == summoner_id]
    actual_status = summoner_row['binary_time_group'].values[0]  # Store the actual status
    
    # Make prediction using the model
    features_for_prediction = summoner_row[feature_columns]
    features_scaled = scaler.transform(features_for_prediction)  # Scale the features
    prediction = model.predict(features_scaled)
    predicted_status = 'Active' if prediction[0] == 1 else 'Inactive'
    
    # Display predictions
    st.write(f'The predicted activity status is: {predicted_status}')
    st.write(f'The actual activity status is: {actual_status}')

    # Filter raw data for selected summoner_id and ensure there's data
    summoner_history = raw_df[raw_df['summoner_id'] == summoner_id]
    
    # Plotting kills and assists over the last few matches if data exists
    if not summoner_history.empty:
        # Plot: Kills and Assists Over Time
        fig, ax = plt.subplots()
        ax.plot(summoner_history['game_creation_dt'], summoner_history['kills'], label='Kills')
        ax.plot(summoner_history['game_creation_dt'], summoner_history['assists'], label='Assists')
        ax.set_xlabel('Game Date')
        ax.set_ylabel('Count')
        ax.set_title('Kills and Assists Over Time')
        ax.legend()
        st.pyplot(fig)

        # Plot: Game Duration vs Kills/Assists
        fig, ax = plt.subplots()
        ax.scatter(summoner_history['game_duration'], summoner_history['kills'], label='Kills', color='blue', alpha=0.6)
        ax.scatter(summoner_history['game_duration'], summoner_history['assists'], label='Assists', color='orange', alpha=0.6)
        ax.set_xlabel('Game Duration')
        ax.set_ylabel('Count')
        ax.set_title('Game Duration vs. Kills/Assists')
        ax.legend()
        st.pyplot(fig)

        # Plot: Gold Earned vs Total Damage Dealt
        fig, ax = plt.subplots()
        ax.scatter(summoner_history['gold_earned'], summoner_history['total_damage_dealt'], label='Damage Dealt', color='green', alpha=0.6)
        ax.set_xlabel('Gold Earned')
        ax.set_ylabel('Total Damage Dealt')
        ax.set_title('Gold Earned vs. Total Damage Dealt')
        st.pyplot(fig)

        # Bar Plot: Game Mode Distribution
        game_mode_counts = summoner_history['game_mode'].value_counts()
        fig, ax = plt.subplots()
        game_mode_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Game Mode')
        ax.set_ylabel('Number of Games')
        ax.set_title('Game Mode Distribution')
        st.pyplot(fig)
        
    else:
        st.write("No historical data available for the selected summoner.")
