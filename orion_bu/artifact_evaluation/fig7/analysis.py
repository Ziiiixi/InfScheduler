import re
from datetime import datetime, timedelta

# Function to parse log lines
def parse_log_line(line):
    pattern = re.compile(
        r'Thread Id: (\d+) Start time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{9}) End time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{9})')
    match = pattern.search(line)
    if match:
        thread_id = int(match.group(1))
        start_time_str = match.group(2)
        end_time_str = match.group(3)

        # Extract seconds and nanoseconds
        start_time_seconds = start_time_str[:19]
        start_time_nanoseconds = start_time_str[20:]
        end_time_seconds = end_time_str[:19]
        end_time_nanoseconds = end_time_str[20:]

        # Parse the seconds part and add nanoseconds
        start_time = datetime.strptime(start_time_seconds, '%Y-%m-%d %H:%M:%S')
        start_time += timedelta(microseconds=int(start_time_nanoseconds) // 1000)

        end_time = datetime.strptime(end_time_seconds, '%Y-%m-%d %H:%M:%S')
        end_time += timedelta(microseconds=int(end_time_nanoseconds) // 1000)

        return thread_id, start_time, end_time
    return None

# Function to check for overlaps
def check_overlaps(events):
    overlaps = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            id1, start1, end1 = events[i]
            id2, start2, end2 = events[j]
            if id1 != id2 and (start1 < end2 and end1 > start2):
                overlaps.append((id1, id2, start1, end1, start2, end2))
    return overlaps

# Read the log file and parse it
def analyze_log(file_path):
    events = []
    with open(file_path, 'r') as file:
        for line in file:
            result = parse_log_line(line)
            if result:
                events.append(result)
    
    # Check for overlaps
    overlaps = check_overlaps(events)
    
    # Print results
    if overlaps:
        print("Overlaps detected:")
        for overlap in overlaps:
            id1, id2, start1, end1, start2, end2 = overlap
            print(f"Thread {id1} ({start1} to {end1}) overlaps with Thread {id2} ({start2} to {end2})")
    else:
        print("No overlaps detected.")

# Replace '123.log' with the path to your log file
analyze_log('123.log')
