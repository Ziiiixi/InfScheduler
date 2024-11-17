import re

# Open and read the log file
with open('123.log', 'r', encoding='ISO-8859-1') as file:
    log_data = file.readlines()

# Regular expression to match the p50, p95, p99 values
pattern = re.compile(r'Client (\d+) finished! p50: ([\d.]+) sec, p95: ([\d.]+) sec, p99: ([\d.]+) sec, average: ([\d.]+)')

# Initialize lists to store the values for each client
p50_values = [None] * 4  # Assuming there are 4 clients (0 to 3)
p95_values = [None] * 4
p99_values = [None] * 4
avg_values = [None] * 4

# Process each line in the log data
for line in log_data:
    match = pattern.search(line)
    if match:
        client_id = int(match.group(1))  # Extract the client ID as an integer
        p50 = match.group(2)
        p95 = match.group(3)
        p99 = match.group(4)
        avg = match.group(5)
        # Store the values in the corresponding index for each client
        p50_values[client_id] = p50
        p95_values[client_id] = p95
        p99_values[client_id] = p99
        avg_values[client_id] = avg

# Print the results in the desired format
print(",".join(p50_values))  # p50 values for clients in order
print(",".join(p95_values))  # p95 values for clients in order
print(",".join(p99_values))  # p99 values for clients in order
print(",".join(avg_values))  # p99 values for clients in order
