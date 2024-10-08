import pandas as pd
import numpy as np

def generate_noisy_covariate(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    deceased_data = df['Deceased'].values[:7200]
    
    noise = np.abs(np.random.normal(0, 0.00001, size=deceased_data.shape))
    
    noisy_deceased = deceased_data + noise
    
    noisy_df = pd.DataFrame({
        'x': noisy_deceased
    })
    
    noisy_df.to_csv(output_csv, index=False)
    print(f"Generated noisy deceased data saved to {output_csv}")

input_csv = "/home/zhicao/ODE/data/weekly_10_data.csv"
output_csv = "/home/zhicao/ODE/data/weekly_10_covariate.csv"
generate_noisy_covariate(input_csv, output_csv)
