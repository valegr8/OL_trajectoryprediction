ol_file_path = "/home/vgrwbx/workspace/OL_trajectoryprediction/val_metrics.csv"
no_ol_path = "/home/vgrwbx/workspace/OL_trajectoryprediction/metrics_files/val_metrics.csv"

threshold = 0.001 # Define your threshold here

ids_with_significant_difference = set()

# Open files again and compare metrics for common IDs
with open(ol_file_path, "r") as file_b, open(no_ol_path, "r") as file_a:
    next(file_a)  # Skip header in file a
    next(file_b)  # Skip header in file b

    for line_b in file_b:
        parts_b = line_b.strip().split(",")

        for line_a in file_a:
            parts_a = line_a.strip().split(",")
            # Check if the scenario ID is in the set of common IDs
            if parts_a[0] == parts_b[0]:
                metrics_a = [float(metric) for metric in parts_a[1:-1]]
                metrics_b = [float(metric) for metric in parts_b[1:-1]]

                # Check if the absolute difference between corresponding metrics exceeds the threshold
                if any(abs(a - b) > threshold for a, b in zip(metrics_a, metrics_b)):
                    ids_with_significant_difference.add(parts_a[0])
                    print(parts_a[0])
                break  # Exit the inner loop once a match is found

# # Print the unique scenario IDs with significant metric differences
# for scenario_id in ids_with_significant_difference:
#     print(scenario_id)