import pandas as pd

# 读取daily data
input_file = '/home/zhicao/ODE/data/daily_10_data.csv'
output_file = '/home/zhicao/ODE/data/weekly_10_data.csv'

# 每周的天数
days_per_week = 7
total_weeks = 72

# 读取数据
df = pd.read_csv(input_file)

# 假设数据结构是按照顺序排列的，每500天为一个周期，共100个周期
num_cycles = 100
days_per_cycle = 500

# 初始化列表来存储转换后的数据
weekly_data = []

# 处理每个周期的数据
for cycle in range(num_cycles):
    start_index = cycle * days_per_cycle
    end_index = start_index + days_per_cycle
    
    # 选取当前周期的数据
    daily_cycle_data = df.iloc[start_index:end_index]

    # 将每日数据转换为每周数据
    for week in range(total_weeks):
        # 计算当前周的起始和结束索引
        week_start = week * days_per_week
        week_end = week_start + days_per_week
        
        # 选取当前周的数据
        daily_week_data = daily_cycle_data.iloc[week_start:week_end]
        
        # 计算beta和Ca的平均值
        beta_avg = daily_week_data['beta'].mean()
        ca_avg = daily_week_data['Ca'].mean()

        # 其余列取和
        weekly_sum = daily_week_data.drop(columns=['beta', 'Ca', 'Time']).sum()
        
        # 创建一行新的周数据
        week_data = {
            'time': week + cycle * total_weeks,  # 将Week列替换为Time
            'beta': beta_avg,
            'Ca': ca_avg,
            'Susceptible': weekly_sum['Susceptible'],
            'Exposed': weekly_sum['Exposed'],
            'Infectious_asymptomatic': weekly_sum['Infectious_asymptomatic'],
            'Infectious_pre-symptomatic': weekly_sum['Infectious_pre-symptomatic'],
            'Infectious_mild': weekly_sum['Infectious_mild'],
            'Infectious_severe': weekly_sum['Infectious_severe'],
            'Hospitalized_recovered': weekly_sum['Hospitalized_recovered'],
            'Hospitalized_deceased': weekly_sum['Hospitalized_deceased'],
            'Recovered': weekly_sum['Recovered'],
            'Deceased': weekly_sum['Deceased'],
        }
        
        # 添加到列表中
        weekly_data.append(week_data)

# 将数据转换为DataFrame
weekly_df = pd.DataFrame(weekly_data)

# 计算a列
weekly_df['a'] = 0  # 初始化a列
for i in range(len(weekly_df) - 1):
    if weekly_df.loc[i, 'beta'] != weekly_df.loc[i + 1, 'beta']:
        weekly_df.loc[i, 'a'] = 1
weekly_df.loc[len(weekly_df) - 1, 'a'] = 0

# 保存到CSV文件
weekly_df.to_csv(output_file, index=False)

print(f'Weekly data saved to {output_file}')
