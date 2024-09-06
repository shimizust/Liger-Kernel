import os
import json
import glob
import statistics
import csv

# Define the directory containing the log files
script_directory = os.path.dirname(os.path.abspath(__file__))
log_directory = os.path.join(script_directory, "results")
output_file = os.path.join(script_directory, "results/benchmark_aggregated_results.csv")

# Get all log files
log_files = glob.glob(os.path.join(log_directory, "*.log"))

# Dict of model -> provider -> batch size -> data
all_data = {}

# Filename like "mistral_use_liger_True_batch_size_64_rep_3.log"
MODEL_TYPE_INDEX = 0
USE_LIGER_INDEX = 3
BATCH_SIZE_INDEX = 6

TOTAL_PEAK_MEMORY_ALLOCATED_MB = "total_peak_memory_allocated_MB"
AVG_TOKENS_PER_SECOND = "avg_tokens_per_second"
OOM_VALUE = "OOM"
LIGER_PROVIDER = "liger"
HUGGINGFACE_PROVIDER = "huggingface"

# Loop over each log file
for log_file in log_files:

    with open(log_file, "r") as file:
        # Extract benchmark details from log filename
        filename = os.path.splitext(os.path.basename(log_file))[0]
        model = filename.split("_")[MODEL_TYPE_INDEX]
        use_liger = filename.split("_")[USE_LIGER_INDEX]
        provider = LIGER_PROVIDER if use_liger == "True" else HUGGINGFACE_PROVIDER
        batch_size = filename.split("_")[BATCH_SIZE_INDEX]

        if model not in all_data:
            all_data[model] = {}
        if provider not in all_data[model]:
            all_data[model][provider] = {}
        if batch_size not in all_data[model][provider]:
            all_data[model][provider][batch_size] = {
                TOTAL_PEAK_MEMORY_ALLOCATED_MB: [],
                AVG_TOKENS_PER_SECOND: []
            }

        lines = file.readlines()
        if lines:
            # Read the last line
            last_line = lines[-1].strip()
            
            try:
                # Parse the last line as a JSON-like dictionary
                data = json.loads(last_line.replace("'", '"'))
                
                # If run fails due to OOM, the last line logged will not have these keys
                total_peak_memory = data.get(TOTAL_PEAK_MEMORY_ALLOCATED_MB, OOM_VALUE)
                avg_tokens_per_second = data.get(AVG_TOKENS_PER_SECOND, OOM_VALUE)

                # Append extracted data to the list
                print(f"Model: {model}, Provider: {provider}, Batch Size: {batch_size}, Total Peak Memory: {total_peak_memory}, Avg Tokens Per Second: {avg_tokens_per_second}")
                all_data[model][provider][batch_size][TOTAL_PEAK_MEMORY_ALLOCATED_MB].append(total_peak_memory)
                all_data[model][provider][batch_size][AVG_TOKENS_PER_SECOND].append(avg_tokens_per_second)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from last line of {log_file}")
                pass

print(json.dumps(all_data, indent=4))

# Calculate the mean/std of the data
summary_stats = []
for model, providers in all_data.items():
    for provider, batch_sizes in providers.items():
        for batch_size, data in batch_sizes.items():
            total_peak_memory_allocated_MB_values = data[TOTAL_PEAK_MEMORY_ALLOCATED_MB]
            avg_tokens_per_second_values = data[AVG_TOKENS_PER_SECOND]

            if len(total_peak_memory_allocated_MB_values) > 1 and len(avg_tokens_per_second_values) > 1:
                if OOM_VALUE in total_peak_memory_allocated_MB_values:
                    total_peak_memory_allocated_MB_mean, total_peak_memory_allocated_MB_std, \
                     avg_tokens_per_second_mean, avg_tokens_per_second_std = [OOM_VALUE] * 4
                else:
                    total_peak_memory_allocated_MB_mean = statistics.mean(total_peak_memory_allocated_MB_values)
                    total_peak_memory_allocated_MB_std = statistics.stdev(total_peak_memory_allocated_MB_values)
                    avg_tokens_per_second_mean = statistics.mean(avg_tokens_per_second_values)
                    avg_tokens_per_second_std = statistics.stdev(avg_tokens_per_second_values)
                    summary_stats.append({
                        "model": model,
                        "provider": provider,
                        "batch_size": batch_size,
                        "total_peak_memory_allocated_MB_mean": total_peak_memory_allocated_MB_mean,
                        "total_peak_memory_allocated_MB_std": total_peak_memory_allocated_MB_std,
                        "avg_tokens_per_second_mean": avg_tokens_per_second_mean,
                        "avg_tokens_per_second_std": avg_tokens_per_second_std
                    })

with open(output_file, mode="w", newline='') as file:
    writer = csv.DictWriter(file, fieldnames=[
        "model", "provider", "batch_size", "total_peak_memory_allocated_MB_mean", 
        "total_peak_memory_allocated_MB_std", "avg_tokens_per_second_mean", "avg_tokens_per_second_std"
    ])
    
    writer.writeheader()  # Write the header row
    writer.writerows(summary_stats)  # Write all rows from summary_stats

print(f"Compiled data has been written to {output_file}")