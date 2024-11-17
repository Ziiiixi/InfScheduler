import pandas as pd

# Load the CSV file
df = pd.read_csv('mobilenetv2_32_fwd')

# Update SM_usage to 10 for all rows except the specific kernel condition
# df.loc[~((df['Name'] == 'ampere_sgemm_32x32_sliced1x4_nn') & 
#          (df['Grid'] == 126) & 
#          (df['Block'] == 128)), 'Knee_TPC'] = 10
# Load the CSV file into a DataFrame

if 'Knee_TPC' not in df.columns:
    print("error")
    df['Knee_TPC'] = 0 

df['Knee_TPC'] = 24
# df.loc[df['Knee_TPC'] <= 8, 'Knee_TPC'] = 8
# df.loc[0:300, 'Knee_TPC'] = 24
# df.loc[(df.index >= 0) & (df.index <= 2000) & (df['Name'].str.contains('ampere')), 'Knee_TPC'] = 8
# df.loc[(df.index >= 0) & (df.index <= 2000) & (~df['Name'].str.contains('ampere')), 'Knee_TPC'] = 12


# Save the modified dataframe back to a CSV file
df.to_csv('mobilenetv2_32_fwd_updated', index=False)
