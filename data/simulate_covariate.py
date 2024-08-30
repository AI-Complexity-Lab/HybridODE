import pandas as pd
import numpy as np

file_path = '/home/zhicao/ODE/data/result.csv'
df = pd.read_csv(file_path)

# 提取前1000天的数据
first_1000_days = df.loc[:999, 'Deceased'].copy()

# 将数据向后平移2天
shifted_data = first_1000_days.shift(-2)

# 用第1001天和第1002天的数据填补最后两天的空缺
shifted_data.iloc[-2:] = df.loc[1000:1001, 'Deceased'].values

# 添加噪声
noise = np.random.normal(0, 0.01, size=shifted_data.shape)
noisy_data = shifted_data + noise

# 保存到新的CSV文件中
output_path = '/home/zhicao/ODE/data/noisy_deceased1.csv'
noisy_data.to_csv(output_path, index=False)

