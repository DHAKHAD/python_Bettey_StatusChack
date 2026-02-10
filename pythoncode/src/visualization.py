import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns

def compute_dq_dv(Q, V):
    """Compute incremental capacity dQ/dV while avoiding division by zero."""
    dV = np.gradient(V)
    dQ = np.gradient(Q)
    dV[dV == 0] = np.nan  
    dq_dv = dQ / dV
    return np.nan_to_num(dq_dv)  

def plot_ica(voltage, dq_dv, test_id):
    """Plot dQ/dV vs. Voltage for different tests."""
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=voltage, y=dq_dv, hue=test_id, palette="coolwarm")
    plt.xlabel("Voltage (V)")
    plt.ylabel("dQ/dV")
    plt.title("Incremental Capacity Analysis")
    plt.legend(title="Test ID")
    plt.show()

def plot_3d_impedance(df):
    """3D Scatter Plot of Impedance vs. Cycle Count."""
    df['Re'] = pd.to_numeric(df['Re'], errors='coerce')
    df['Rct'] = pd.to_numeric(df['Rct'], errors='coerce')
    df['test_id'] = pd.to_numeric(df['test_id'], errors='coerce')
    df = df.dropna(subset=['Re', 'Rct', 'test_id'])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Re'], df['Rct'], df['test_id'], c=df['test_id'], cmap='viridis')
    ax.set_xlabel('Real Impedance (Re)')
    ax.set_ylabel('Charge Transfer Resistance (Rct)')
    ax.set_zlabel('Test ID')
    plt.colorbar(scatter, label="Test ID")
    plt.title("Impedance vs. Test ID")
    plt.show()

if __name__ == "__main__":
    file_path = "./data/processed_battery_data.csv"
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        numeric_cols = ['Capacity', 'Re', 'Rct', 'test_id']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') 
        impedance_df = df[df["type"] == "impedance"].dropna(subset=['Re', 'Rct'])
        if 'Capacity' in df.columns and df['Capacity'].notna().sum() > 0:
            voltage = np.linspace(3.0, 4.2, len(df)) 
            df['dQ/dV'] = compute_dq_dv(df['Capacity'], voltage)
            plot_ica(voltage, df['dQ/dV'], df['test_id'])
        if not impedance_df.empty:
            plot_3d_impedance(impedance_df)
        else:
            print(" No valid impedance data found for plotting.")

    except FileNotFoundError:
        print(f" File not found: {file_path}")
    except KeyError as e:
        print(f"Missing column: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
