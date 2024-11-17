import re
import csv
from fuzzywuzzy import fuzz, process

def parse_log_file_1(log_file):
    kernel_data = []

    with open(log_file, 'r') as file:
        lines = file.readlines()
        kernel_name = None
        execution_time = None

        for line in lines:
            if line.startswith("name of kernel is"):
                # Extract the kernel name using regex
                match = re.search(r'name of kernel is (.*)', line)
                if match:
                    kernel_name = match.group(1).strip()
            elif "execution time:" in line:
                # Extract the execution time using regex
                match = re.search(r'execution time:\s+([\d\.]+)\s+ns', line)
                if match:
                    execution_time = float(match.group(1).strip())
                
                # Store the kernel name and execution time if both are found
                if kernel_name and execution_time is not None:
                    kernel_data.append((kernel_name, execution_time))
                    # Reset for the next entry
                    kernel_name = None
                    execution_time = None

    return kernel_data

def parse_log_file_2(log_file):
    kernel_data = []

    with open(log_file, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split(',')
        # Find the index of the relevant columns
        name_index = header.index('Name')
        grid_index = header.index('Grid')
        block_index = header.index('Block')

        for line in lines[1:]:
            columns = line.strip().split(',')
            kernel_name = columns[name_index].strip()
            grid_size = columns[grid_index].strip()
            block_size = columns[block_index].strip()
            if kernel_name and grid_size and block_size:
                # Store kernel name, grid size, and block size
                kernel_data.append((kernel_name, grid_size, block_size))

    return kernel_data


def match_kernel_names(kernel_data_1, kernel_data_2):
    # Convert kernel_data_2 to a dictionary of lists of (grid_size, block_size) tuples
    kernel_data_2_dict = {}
    for kernel_name, grid_size, block_size in kernel_data_2:
        if kernel_name not in kernel_data_2_dict:
            kernel_data_2_dict[kernel_name] = []
        kernel_data_2_dict[kernel_name].append((grid_size, block_size))

    matched_data = []
    unmatched_kernels = set()

    for kernel_name_1, execution_time in kernel_data_1:
        if kernel_name_1 in kernel_data_2_dict and kernel_data_2_dict[kernel_name_1]:
            # Get the grid and block sizes for this kernel name
            grid_size, block_size = kernel_data_2_dict[kernel_name_1].pop(0)  # Use the first available pair
            matched_data.append((kernel_name_1, grid_size, block_size, execution_time))
        else:
            unmatched_kernels.add(kernel_name_1)
            print(f"Error: Kernel name '{kernel_name_1}' not found or no suitable match found in the second log file.")

    return matched_data, unmatched_kernels



def save_to_csv(matched_data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['Kernel Name', 'Grid Size', 'Block Size', 'Execution Time (ms)'])
        # Write the matched data
        for kernel_name, grid_size, block_size, execution_time in matched_data:
            csvwriter.writerow([kernel_name, grid_size, block_size, execution_time])

# Example usage
log_file_1 = "123.log"
log_file_2 = "/home/zixi/orion/benchmarking/model_kernels/resnet152_16_fwd" #mmy
# log_file_2 = "/home/zixi/orion/benchmarking/model_kernels/resnet152_32_fwd"
output_file = "combined_kernel_data.csv"

kernel_data_1 = parse_log_file_1(log_file_1)
print(len(kernel_data_1))
kernel_data_2 = parse_log_file_2(log_file_2)
print(len(kernel_data_2))
matched_data, unmatched_kernels = match_kernel_names(kernel_data_1, kernel_data_2)
save_to_csv(matched_data, output_file)

print(f"Combined data has been written to {output_file}")



