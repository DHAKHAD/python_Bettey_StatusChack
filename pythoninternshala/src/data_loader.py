import os
import numpy as np
import pandas as pd
from ast import literal_eval

def load_battery_data(filepath):
    """Load NASA battery dataset from a CSV file and clean the data."""
    try:
        
        df = pd.read_csv(filepath)
        print(" Loaded as CSV file")
    except Exception as e:
        print(f" Error loading file: {e}")
        return None

    
    def parse_time(time_str):
        try:
            time_list = literal_eval(time_str)  
            return pd.Timestamp(year=int(time_list[0]), 
                                month=int(time_list[1]), 
                                day=int(time_list[2]), 
                                hour=int(time_list[3]), 
                                minute=int(time_list[4]), 
                                second=int(float(time_list[5])))  
        except:
            return np.nan  

    df["start_time"] = df["start_time"].apply(parse_time)

    
    df.fillna(method="ffill", inplace=True)

    return df

if __name__ == "__main__":
    filepath = "./data/Battery_Data.csv"
    
    if not os.path.exists(filepath):
        print(" File not found! Check the path.")
    else:
        df = load_battery_data(filepath)
        
        if df is not None:
            df.to_csv("./data/processed_battery_data.csv", index=False)
            print(" Data saved to processed_battery_data.csv")
            print(df.head())
        else:
            print("Failed to load data. Please check dataset format.")
