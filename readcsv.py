import pandas as pd

def readcsv(filename=''):
    df = pd.read_csv(filename)
    rx = df['Voltage (V)'].values
    return rx