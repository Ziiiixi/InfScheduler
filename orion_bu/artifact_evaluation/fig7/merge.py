import pandas as pd
# Define the path to your input file and output file
input_file = 'combined_kernel_data.csv'
output_file = 'full_kernel_data_with_averages.csv'
# Load the CSV file into a DataFrame
df = pd.read_csv(input_file)
# Group by 'Kernel Name', 'Grid Size', and 'Block Size' and calculate the average execution time
grouped = df.groupby(['Kernel Name', 'Grid Size', 'Block Size'], as_index=False)
# Calculate the mean of the 'Execution Time (ms)' and retain all other columns in the group
aggregated_data = grouped.agg({
    'Execution Time (ms)': 'mean',
    # If you have other numeric columns you want to aggregate, add them here:
    # 'SomeOtherColumn': 'mean'
})
# Optionally, if you want to keep other non-aggregated columns (like the first instance of each group):
# you can use .first() to grab the first non-numeric columns for each group.
first_instance_columns = grouped.first()
# Merge the aggregated data with the first instance columns
result = pd.merge(first_instance_columns, aggregated_data, on=['Kernel Name', 'Grid Size', 'Block Size'], suffixes=('', '_avg'))
# Write the result to the output file
result.to_csv(output_file, index=False)
print(f'Full kernel data with average execution times has been saved to {output_file}')