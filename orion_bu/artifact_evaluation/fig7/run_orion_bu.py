import os
import time
import argparse
import pandas as pd
num_runs = 1
# trace_files = [
#     # ("", "", "profile", 160000),
#     ("", "", "5percent/Mnet_8_Rnet_32_Mnet_16_Rnet_8", 160000),
#     # ("", "", "5percent/Mnet_16_Mnet_8_Rnet_32_Rnet_16", 160000),
#     # ("", "", "5percent/Mnet_16_Rnet_16_Mnet_32_Rnet_32", 160000),
#     # ("", "", "5percent/Mnet_32_Rnet_8_Mnet_8_Rnet_16", 160000),
#     # ("", "", "5percent/Rnet_8_Mnet_8_Rnet_16_Mnet_16", 160000),
#     # ("", "", "5percent/Rnet_8_Mnet_32_Rnet_32_Mnet_16", 160000),
#     # ("", "", "5percent/Rnet_8_Rnet_8_Mnet_32_Mnet_16", 160000),
#     # ("", "", "5percent/Rnet_32_Mnet_32_Rnet_8_Mnet_16", 160000),
# ]
trace_files = [
    ("", "", "Mnet_8_Rnet_32_Mnet_16_Rnet_8", 160000),
]

base_path = "/home/zixi/orion_bu/profiling/benchmarks"
output_base_path = "/home/zixi/orion_bu/benchmarking/model_kernels/torun"

def map_model_name(model):
    if model == "Mnet":
        return "mobilenetv2"
    elif model == "Rnet":
        return "resnet152"
    else:
        return model

def update_knee_tpc_for_trace(model_dir):
    file_path = os.path.join(model_dir, "kernel_analysis_with_best_tpc.csv")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    knee_tpc_dict = {}
    
    for idx, row in df.iterrows():
        tpc_values = row[4:28].values 
        knee_tpc = 24 
        for i in range(23, 0, -1):
            if tpc_values[i-1] > 0:  
                reduce = (tpc_values[i-1] - tpc_values[i]) / tpc_values[i] * 100
                if reduce <= 5 or tpc_values[i-1] - tpc_values[i] < 1000:
                    knee_tpc = i + 1  
                else:
                    break  
        
        key = (row['Kernel Name'], row['Grid Size'], row['Block Size'])
        knee_tpc_dict[key] = knee_tpc
    
    # Store performance for global top 10 selection
    df['Performance'] = df.iloc[:, 24]  # Assuming column 24 is 0-indexed at 24
    return knee_tpc_dict, df[['Kernel Name', 'Grid Size', 'Block Size', 'Performance']]

global_kernels = []

# Process each trace file to collect performance data
for trace in trace_files:
    trace_name = trace[2] 
    parts = trace_name.split("_")
    for i in range(0, len(parts), 2):
        model = parts[i] 
        batch_size = parts[i + 1]  
        mapped_model = map_model_name(model)
        model_dir = os.path.join(base_path, f"{mapped_model.lower()}_bz{batch_size}") 
        
        knee_tpc_dict, performance_data = update_knee_tpc_for_trace(model_dir)
        
        if knee_tpc_dict is not None:
            global_kernels.append(performance_data)

# Combine all performance data to find the top 10 expensive kernels
if global_kernels:
    combined_performance = pd.concat(global_kernels)
    combined_performance_sorted = combined_performance.sort_values(by='Performance', ascending=False)
    top_expensive_kernels = combined_performance_sorted.head(50)
    # print("Top 10 Most Expensive Kernels:")
    # print(top_expensive_kernels)
    global_critical_kernels = set(zip(top_expensive_kernels['Kernel Name'], top_expensive_kernels['Grid Size'], top_expensive_kernels['Block Size']))

# Now update the fwd CSV files using the global critical kernels
def add_knee_tpc_to_fwd_csv(model_dir, model_name, batch_size, knee_tpc_dict):
    fwd_file_path = os.path.join(model_dir, f"{model_name}_{batch_size}_fwd")
    
    if not os.path.exists(fwd_file_path):
        print(f"Forward file not found: {fwd_file_path}")
        return
    
    fwd_df = pd.read_csv(fwd_file_path)
    fwd_df['Knee_TPC'] = None
    fwd_df['Is_Critical'] = 0  # Initialize with 0
    for idx, fwd_row in fwd_df.iterrows():
        key = (fwd_row['Name'], fwd_row['Grid'], fwd_row['Block'])
        if key in knee_tpc_dict:
            fwd_df.at[idx, 'Knee_TPC'] = knee_tpc_dict[key]
        else:
            print(key)
            print(knee_tpc_dict)
            print("error")
            exit()

        if key in global_critical_kernels:
            fwd_df.at[idx, 'Is_Critical'] = 1  # Mark as critical
    
    # Save the updated forward DataFrame to the new output path
    fwd_output_path = os.path.join(output_base_path, f"{model_name}_{batch_size}_fwd_updated")
    fwd_df.to_csv(fwd_output_path, index=False)
    print(f"Updated forward CSV saved at: {fwd_output_path}")

# Update fwd files using the collected global critical kernels
for trace in trace_files:
    trace_name = trace[2] 
    parts = trace_name.split("_")
    for i in range(0, len(parts), 2):
        model = parts[i] 
        batch_size = parts[i + 1]  
        mapped_model = map_model_name(model)
        model_dir = os.path.join(base_path, f"{mapped_model.lower()}_bz{batch_size}") 
        
        knee_tpc_dict, performance_data = update_knee_tpc_for_trace(model_dir)
        
        if knee_tpc_dict is not None:
            add_knee_tpc_to_fwd_csv(model_dir, mapped_model, batch_size, knee_tpc_dict)  


# for (be, hp, f, max_be_duration) in trace_files:
#     for run in range(num_runs):
#         print(be, hp, run, flush=True)
#         # run
#         file_path = f"config_files/baselines/{f}.json"
#         print(file_path)
#         # print(mymask)
#         os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion_bu/src/cuda_capture/libinttemp.so' python3.8 ../../benchmarking/launch_jobs.py --algo orion --config_file {file_path} --orion_max_be_duration {max_be_duration}")
#         # copy results
#         os.system(f"cp client_1.json results/orion_bu/{be}_{hp}_{run}_hp.json")
#         os.system(f"cp client_0.json results/orion_bu/{be}_{hp}_{run}_be.json")

#         os.system("rm client_1.json")
#         os.system("rm client_0.json")