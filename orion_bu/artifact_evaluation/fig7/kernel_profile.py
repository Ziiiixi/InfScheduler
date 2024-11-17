import re
import pandas as pd

# Regex patterns to capture necessary information
start_pattern = r"Name of kernel: (?P<kernel_name>[\w\:\s\.\|]+) \| Client ID: (?P<client_id>\d+) \| Iteration: (?P<iteration>\d+) \| Kernel Index: (?P<kernel_idx>\d+) \| Start Time: (?P<start_time>\d+) ns"
end_pattern = r"Kernel (?P<kernel_idx>\d+) in iteration (?P<iteration>\d+) from idx (?P<client_id>\d+) end at time (?P<end_time>\d+) ns"

# Initialize dictionaries to store kernel data and results for each client
data = {'client_0': [], 'client_1': []}
active_kernels = {}  # Stores kernels that have started but not yet finished

# Read the log file and process line by line
with open('123.log', 'r') as file:
    for line in file:
        # Check if the line is a start event
        start_match = re.match(start_pattern, line)
        if start_match:
            kernel_name = start_match.group("kernel_name").strip()
            client_id = int(start_match.group("client_id"))
            iteration = int(start_match.group("iteration"))
            kernel_idx = int(start_match.group("kernel_idx"))
            start_time = int(start_match.group("start_time"))
            
            # Store the start details in active_kernels
            active_kernels[(client_id, iteration, kernel_idx)] = {
                'Kernel Name': kernel_name,
                'Iteration': iteration,
                'Kernel Index': kernel_idx,
                'Start Time': start_time
            }

        # Check if the line is an end event
        end_match = re.match(end_pattern, line)
        if end_match:
            client_id = int(end_match.group("client_id"))
            iteration = int(end_match.group("iteration"))
            kernel_idx = int(end_match.group("kernel_idx"))
            end_time = int(end_match.group("end_time"))

            # Retrieve the start info and calculate the duration
            key = (client_id, iteration, kernel_idx)
            if key in active_kernels:
                kernel_info = active_kernels.pop(key)
                start_time = kernel_info['Start Time']
                duration = end_time - start_time
                
                # Add duration and end time to the kernel info
                kernel_info['End Time'] = end_time
                kernel_info['Duration (ns)'] = duration

                # Append to the respective client list
                if client_id == 0:
                    data['client_0'].append(kernel_info)
                elif client_id == 1:
                    data['client_1'].append(kernel_info)

# Convert to DataFrames and write to CSV files
df_client_0 = pd.DataFrame(data['client_0'])
df_client_1 = pd.DataFrame(data['client_1'])

# Write each client's DataFrame to a separate CSV file
df_client_0.to_csv('client_0_kernels.csv', index=False)
df_client_1.to_csv('client_1_kernels.csv', index=False)

print("Data saved to client_0_kernels.csv and client_1_kernels.csv")
