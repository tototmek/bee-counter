import csv

file_path = "data/experiments/2025-07-26/2025-07-26_11-04-07.csv"

time_column = []

with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        # Check if the row is not empty to avoid errors
        if row and row[0].isnumeric():
            time_column.append(int(row[0]))

time_span = (time_column[-1]-time_column[0]) / 60000
n_samples = len(time_column)

print(f"Experiment took {time_span} minutes and generated {n_samples} samples.")