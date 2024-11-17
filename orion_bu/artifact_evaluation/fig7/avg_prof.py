import pandas as pd

# Define the path to your input file and output file
input_file = 'combined_kernel_data.csv'
output_file = 'average_execution_times.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Group by 'Kernel Name', 'Grid Size', and 'Block Size' and calculate the average execution time
# Also, count the number of instances for each group
grouped = df.groupby(['Kernel Name', 'Grid Size', 'Block Size'])
average_times = grouped['Execution Time (ms)'].mean().reset_index()
instance_counts = grouped.size().reset_index(name='Instance Count')

# Merge the average execution times with the instance counts
# result = pd.merge(average_times, instance_counts, on=['Kernel Name', 'Grid Size', 'Block Size'])
average_times = df.groupby(['Kernel Name', 'Grid Size', 'Block Size'])['Execution Time (ms)'].mean()

# Write only the average execution times to the output file
average_times.to_csv(output_file, header=['Average Execution Time (ms)'], index=False)

# Write the result to the output file
# average_times.to_csv(output_file, index=False)

print(f'Average execution times with counts have been saved to {output_file}')
