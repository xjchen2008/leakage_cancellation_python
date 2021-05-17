import pandas as pd
import matplotlib.pyplot as plt

def readcsv(filename=''):
    df = pd.read_csv(filename)
    rx = df['Voltage (V)'].values
    return rx

if __name__ == '__main__':
    filename ='data\EQ_file_received_chirp_3999_PA_attenna.csv'
    rx = readcsv(filename=filename)
    plt.plot(rx)
    plt.show()