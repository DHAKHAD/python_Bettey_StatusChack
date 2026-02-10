from model_training import train_model
from data_loader import load_battery_data
from visualization import plot_3d_impedance
import pandas as pd
import os

if __name__ == "__main__":
    # Define paths
    data_dir = r"C:\Users\sunil\OneDrive\Desktop\pythoninternshala\data"
    file_path = os.path.join(data_dir, "Battery_Data.csv")
    processed_csv_path = os.path.join(data_dir, "processed_battery_data.csv")

    # Load Data
    df = load_battery_data(file_path)

    if df is not None and not df.empty:
        df.to_csv(processed_csv_path, index=False)
        print(f" Data saved to {processed_csv_path}")
        
        # Visualize 3D Impedance
        plot_3d_impedance(df)

        # Train Model
        model = train_model(df)
    else:
        print(" Failed to load data. Exiting...")
