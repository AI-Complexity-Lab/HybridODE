import pandas as pd
import numpy as np

def generate_noisy_covariate(input_csv, output_csv):
    # 读取输入的CSV文件
    df = pd.read_csv(input_csv)
    
    # 提取前7200个周的死亡人数（"Deceased"）
    deceased_data = df['Deceased'].values[:7200]
    
    # 生成正噪声
    noise = np.abs(np.random.normal(0, 0.00001, size=deceased_data.shape))
    
    # 生成有噪声的死亡人数列
    noisy_deceased = deceased_data + noise
    
    # 创建新的DataFrame保存结果
    noisy_df = pd.DataFrame({
        'x': noisy_deceased  # covariate命名为"x"
    })
    
    # 保存到新的CSV文件
    noisy_df.to_csv(output_csv, index=False)
    print(f"Generated noisy deceased data saved to {output_csv}")

# 调用函数生成新的CSV文件
input_csv = "/home/zhicao/ODE/data/weekly_data_with_treatment.csv"
output_csv = "/home/zhicao/ODE/data/weekly_noisy_deceased.csv"
generate_noisy_covariate(input_csv, output_csv)
