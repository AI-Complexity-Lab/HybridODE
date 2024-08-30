import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
csv_file_path = '/home/zhicao/ODE/result2.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Get the y0_index and Ca value of the first row
first_y0_index = df.iloc[0]['y0_index']
first_ca_value = df.iloc[0]['Ca']

# Filter the data for the same y0_index and Ca as the first row
filtered_data = df[(df['y0_index'] == first_y0_index) & (df['Ca'] == first_ca_value)]

# Separate data for the two beta scenarios
beta_constant_data = filtered_data[filtered_data['beta'] == 0.5].groupby('Time').mean().reset_index()
beta_decay_data = filtered_data[filtered_data['beta'] != 0.5].groupby('Time').mean().reset_index()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(beta_constant_data['Time'], beta_constant_data['Deceased'], label='Beta Constant (0.5)', color='blue')
plt.plot(beta_decay_data['Time'], beta_decay_data['Deceased'], label='Beta Decay', color='red')

# Labels and title
plt.xlabel('Time')
plt.ylabel('Deaths')
plt.title('Deaths Over Time for First y0 with Ca = 0.425')
plt.legend()

# Save the plot as an image file
plt.savefig('/home/zhicao/ODE/deaths_over_time_first_y0_ca_0_425.png')

# Show the plot
plt.show()


