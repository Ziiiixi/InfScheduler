import pandas as pd

file_name = 'resnet152_32_fwd_15percent'
data = pd.read_csv(file_name, header=None, names=[
    'Name', 'Profile', 'Memory_footprint', 'SM_usage', 'Duration', 'Grid', 'Block', 'Knee_TPC'])

data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')
data.dropna(subset=['Duration'], inplace=True)
grouped_data = data.groupby(['Name', 'Grid', 'Block']).agg({'Duration': 'mean'}).reset_index()
top10_groups = grouped_data.nlargest(10, 'Duration')
print(top10_groups)
top10_set = set(top10_groups[['Name', 'Grid', 'Block']].apply(tuple, axis=1))
data['Is_Critical'] = data.apply(
    lambda row: 1 if (row['Name'], row['Grid'], row['Block']) in top10_set else 0, axis=1)


data.to_csv(file_name+'_with_critical_10', index=False)
print("New file with 'Is_Critical' column saved")