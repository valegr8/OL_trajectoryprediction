ol_file_path = "/home/vgrwbx/workspace/OL_trajectoryprediction/val_metrics.csv"
no_ol_path = "/home/vgrwbx/workspace/OL_trajectoryprediction/metrics_files/val_metrics.csv"

from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    # static_map_path: Path to the JSON file containing map data. The file name must match
    # the following pattern: "log_map_archive_{log_id}.json".
    # path to where the logs live
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    ids_with_significant_difference = set()
    count_positive = 0
    count_neg = 0

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
                    if ((float(parts_a[2]) - float(parts_b[2])) >= args.threshold) or ((float(parts_a[4]) - float(parts_b[4])) >= args.threshold):
                        ids_with_significant_difference.add(parts_a[0])
                        count_positive+=1
                        print('ADE:', parts_a[2], '-', parts_b[2], 'FDE: ', parts_a[4],  '-',  parts_b[4], ' ID:', parts_a[0])

                    if ((float(parts_b[2]) - float(parts_a[2])) >= args.threshold) or ((float(parts_b[4]) - float(parts_a[4])) >= args.threshold):
                        count_neg+=1
                    break  # Exit the inner loop once a match is found

    print('IMPROVED: ', count_positive, 'WORSENED: ', count_neg)
    # # Print the unique scenario IDs with significant metric differences
    # for scenario_id in ids_with_significant_difference:
    #     print(scenario_id)