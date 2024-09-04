import os
import json
import glob
import statistics

# Define the directory containing the log files
log_directory = "./results"  # Update with the path to your log files if not in the current directory
output_file = "compiled_results.csv"


# Get all log files
log_files = glob.glob(os.path.join(log_directory, "*.log"))

# Dict of model -> batch size -> data
all_data = {}

# Filename like "mistral_use_liger_True_batch_size_64_rep_3.log"
# Extract model (e.g. mistral)
# Extract batch size (e.g. 64)

# Loop over each log file
for log_file in log_files:
    # Extract the model and batch size from the filename
    filename = os.path.basename(log_file)
    model = filename.split("_")[0]
    batch_size = filename.split("_")[6]

    if model not in all_data:
        all_data = {model: {batch_size: {
            "total_peak_memory_allocated_MB": [],
            "avg_tokens_per_second": []
        }}}
    elif batch_size not in all_data[model]:
        all_data[model][batch_size] = {
            "total_peak_memory_allocated_MB": [],
            "avg_tokens_per_second": []
        }

    with open(log_file, "r") as file:
        lines = file.readlines()
        if lines:
            # Read the last line
            last_line = lines[-1].strip()
            
            try:
                # Parse the last line as a JSON-like dictionary
                data = json.loads(last_line.replace("'", '"'))
                total_peak_memory = data.get("total_peak_memory_allocated_MB")
                avg_tokens_per_second = data.get("avg_tokens_per_second")

                # Extract the run configuration from the filename
                run_config = os.path.basename(log_file).replace(".log", "")

                # Append extracted data to the list
                all_data[model][batch_size]["total_peak_memory_allocated_MB"].append(total_peak_memory)
                all_data[model][batch_size]["avg_tokens_per_second"].append(avg_tokens_per_second)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from last line of {log_file}")

print(json.dumps(all_data, indent=4))

# Calculate the mean/std of the data
summary_stats = []
for model, batch_sizes in all_data.items():
    for batch_size, data in batch_sizes.items():
        total_peak_memory_allocated_MB_values = data["total_peak_memory_allocated_MB"]
        avg_tokens_per_second_values = data["avg_tokens_per_second"]

        if len(total_peak_memory_allocated_MB_values) > 1 and len(avg_tokens_per_second_values) > 1:
            total_peak_memory_allocated_MB_mean = statistics.mean(total_peak_memory_allocated_MB_values)
            total_peak_memory_allocated_MB_std = statistics.stdev(total_peak_memory_allocated_MB_values)
            avg_tokens_per_second_mean = statistics.mean(avg_tokens_per_second_values)
            avg_tokens_per_second_std = statistics.stdev(avg_tokens_per_second_values)
            summary_stats.append({
                "model": model,
                "batch_size": batch_size,
                "total_peak_memory_allocated_MB_mean": total_peak_memory_allocated_MB_mean,
                "total_peak_memory_allocated_MB_std": total_peak_memory_allocated_MB_std,
                "avg_tokens_per_second_mean": avg_tokens_per_second_mean,
                "avg_tokens_per_second_std": avg_tokens_per_second_std
            })

# Write the compiled data to a CSV file
with open(output_file, "w") as file:
    file.write("model,batch_size,total_peak_memory_allocated_MB_mean,total_peak_memory_allocated_MB_std,avg_tokens_per_second_mean,avg_tokens_per_second_std\n")
    for entry in summary_stats:
        file.write(f"{entry['model']},{entry['batch_size']},{entry['total_peak_memory_allocated_MB_mean']},{entry['total_peak_memory_allocated_MB_std']},{entry['avg_tokens_per_second_mean']},{entry['avg_tokens_per_second_std']}\n")

print(f"Compiled data has been written to {output_file}")