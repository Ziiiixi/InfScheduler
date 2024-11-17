import pandas as pd

# Load the datasets
first_data = pd.read_csv('mobilenetv2_8_fwd')
second_data = pd.read_csv('15percent/tpc_15percent_Mobilenetv2_8.csv')

# Prepare to store results
merged_data = []

# Iterate through the first dataset
for _, row in first_data.iterrows():
    name = row['Name']
    grid = row['Grid']
    block = row['Block']
    
    # Find matching rows in the second dataset
    matches = second_data[(second_data['Kernel Name'] == name) &
                          (second_data['Grid Size'] == grid) &
                          (second_data['Block Size'] == block)]
    
    # If matches found, append the data
    if not matches.empty:
        for _, match in matches.iterrows():
            # Combine row with all match columns from 1 to 24
            combined_row = row.tolist() + match[4:28].tolist()  # Columns 1-24 are at index 4 to 27
            merged_data.append(combined_row)

# Create a DataFrame from merged data
# Create column names for the combined DataFrame
new_columns = list(first_data.columns) + list(second_data.columns[4:28])  # Retain first data columns and add match columns
merged_df = pd.DataFrame(merged_data, columns=new_columns)

# Save to a new CSV file
merged_df.to_csv('15percent/mobilenetv2_8_fwd_15percent', index=False)
