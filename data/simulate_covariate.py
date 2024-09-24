import pandas as pd
import numpy as np

file_path = '/home/zhicao/ODE/data/result1.csv'
df = pd.read_csv(file_path)

first_500_days = df.loc[:499, 'Deceased'].copy()

shifted_data = first_500_days.shift(-2)

shifted_data.iloc[-2:] = df.loc[500:501, 'Deceased'].values

noise = np.random.normal(0, 0.01, size=shifted_data.shape)
noisy_data = shifted_data + noise

output_path = '/home/zhicao/ODE/data/noisy_deceased.csv'
noisy_data.to_csv(output_path, index=False)