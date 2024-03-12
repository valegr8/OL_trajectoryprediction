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
        # print(parts_b[0])

        for line_a in file_a:
            parts_a = line_a.strip().split(",")
            
            # Check if the scenario ID is in the set of common IDs
            if parts_a[0] == parts_b[0]:
                metrics_a = {float(parts_a[2]), float(parts_a[4])} #ade,fde 
                metrics_b = {float(parts_b[2]), float(parts_b[4])} #ade,fde 

                # Check if the absolute difference between corresponding metrics exceeds the threshold
                for a, b in zip(metrics_a, metrics_b):
                    if abs(a - b) > threshold:
                        ids_with_significant_difference.add(parts_a[0])
                        print(a, '-', b, ' ID:', parts_a[0])
                        break
                break  # Exit the inner loop once a match is found

# # Print the unique scenario IDs with significant metric differences
# for scenario_id in ids_with_significant_difference:
#     print(scenario_id)