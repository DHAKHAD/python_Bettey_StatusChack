from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  root_mean_squared_error
import pandas as pd
import numpy as np

def train_model(df):
    """Train a RandomForestRegressor to predict Capacity."""
    df['Re'] = pd.to_numeric(df['Re'], errors='coerce')
    df['Rct'] = pd.to_numeric(df['Rct'], errors='coerce')
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')

    
    df = df.dropna(subset=['Re', 'Rct', 'Capacity'])

    # Features & Target
    X = df[['Re', 'Rct']]
    y = df['Capacity']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("Model RMSE:", rmse)
    return model
if __name__ == "__main__":
    file_path = "./data/processed_battery_data.csv"

    try:
        df = pd.read_csv(file_path)

    
        df.columns = df.columns.str.strip()
        numeric_cols = ['Capacity', 'Re', 'Rct']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  
        df = df[df["type"] == "impedance"].dropna(subset=['Re', 'Rct', 'Capacity'])

        if not df.empty:
            trained_model = train_model(df)
        else:
            print(" No valid impedance data found for training.")

    except FileNotFoundError:
        print(f" File not found: {file_path}")
    except KeyError as e:
        print(f" Missing column: {e}")
    except Exception as e:
        print(f" Unexpected error: {e}")
