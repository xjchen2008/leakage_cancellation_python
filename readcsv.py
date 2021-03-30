import pandas as pd
df = pd.read_csv('testplan.csv')
#df.to_string(index = False)
print(df.head())

for i in range(len(df.index)):
    position = df.at[df.index[i], 'angle']
    time = df.at[df.index[i],'time']
    print position, time, len(df.index), df.index[i]
