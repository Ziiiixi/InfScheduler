import re

def accumulate_execution_time(log_file_path):
    total_time = 0.0  # To accumulate the total execution time
    count = 0  # To count the number of entries

    # Regular expression to match the execution time line
    pattern = re.compile(r'Seach Execution time: ([\d.]+) ms')

    # Open the log file with a specific encoding (like 'ISO-8859-1' or 'Windows-1252')
    with open(log_file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract the execution time and convert it to float
                execution_time = float(match.group(1))
                total_time += execution_time
                count += 1

    # Print the results
    if count > 0:
        print(f'Total Execution Time: {total_time:.2f} ms')
        print(f'Average Execution Time: {total_time / count:.2f} ms over {count} entries')
    else:
        print('No execution time entries found.')

# Example usage
log_file_path = '123.log'  # Replace with your .log file path
accumulate_execution_time(log_file_path)
