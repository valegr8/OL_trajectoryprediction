from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import csv

def calculate_percentage_improvement(filtered_dfol, filtered_df):
    difference_ADE = filtered_dfol['val_minADE'] - filtered_df['val_minADE']
    difference_FDE = filtered_dfol['val_minFDE'] - filtered_df['val_minFDE']

    percentage_improvement_ADE = (difference_ADE.mean() / filtered_df['val_minADE'].mean()) * 100
    percentage_improvement_FDE = (difference_FDE.mean() / filtered_df['val_minFDE'].mean()) * 100

    return percentage_improvement_ADE, percentage_improvement_FDE

def calculate_percentage_improvement_sum(filtered_dfol, filtered_df):
    percentage_improvement_ADE = 100 - (filtered_dfol['val_minADE'].sum() / filtered_df['val_minADE'].sum()) * 100
    percentage_improvement_FDE = 100 - (filtered_dfol['val_minFDE'].sum() / filtered_df['val_minFDE'].sum()) * 100

    return percentage_improvement_ADE, percentage_improvement_FDE

def write_to_file(writer, data):
    writer.writerow(data)

def main():
    parser = ArgumentParser()
    parser.add_argument('--log_id', type=str, default="7103fee0-bd5e-4fa3-a8e0-f9753ca1ecf7")
    parser.add_argument('--filepath', type=str, default="/home/vgrwbx/workspace/OL_trajectoryprediction/val_metrics_05.csv")
    parser.add_argument('--noolfilepath', type=str, default="/home/vgrwbx/workspace/OL_trajectoryprediction/metrics_files/val_metrics_nool.csv")
    args = parser.parse_args()

    # Read the files into pandas DataFrames
    df = pd.read_csv(args.noolfilepath)
    df_ol = pd.read_csv(args.filepath)

    # Read scenario IDs from CSV fileabove_threshold_samples
    scenario_ids_df = pd.read_csv("/home/vgrwbx/workspace/OL_trajectoryprediction/metrics_files/above_threshold_samples.csv")
    scenario_ids = scenario_ids_df['scenario_id'].tolist()

    # # Filter data for scenario IDs from the CSV file for EDGE CASE dataset
    df = df[df['scenario_id'].isin(scenario_ids)]
    df_ol = df_ol[df_ol['scenario_id'].isin(scenario_ids)]

    # Get unique scenario IDs
    unique_scenario_ids = df_ol['scenario_id'].unique()

    # # Specify the minimum number of rows required per scenario ID
    # min_rows_per_scenario = 8  # Adjust this number based on your requirement

    # # Filter and process rows by scenario ID
    # for scenario_id, group_df in df.groupby('scenario_id'):
    #     if not len(group_df) >= min_rows_per_scenario:
    #         # Process the rows for this scenario ID (e.g., print or perform operations)
    #         # print(f"Scenario ID: {scenario_id}")
    #         # print(group_df)
    #     # else:
    #         # Delete scenario IDs with fewer rows than the minimum threshold
    #         df = df[df['scenario_id'] != scenario_id]

    # # Specify the path for the output CSV file (filtered data)
    # output_file_path = 'path_to_output_filtered_csv_file.csv'

    # # Save filtered DataFrame to CSV file
    # df.to_csv(output_file_path, index=False)

    # # Drop duplicate rows based on scenario_id and timestep columns
    # df_cleaned = df.drop_duplicates(subset=['scenario_id', 'timestep'], keep='first')

    # # Specify the path for the output CSV file (cleaned data)
    # output_file_path = 'path_to_output_cleaned_csv_file.csv'

    # # Save cleaned DataFrame to CSV file
    # df_cleaned.to_csv(output_file_path, index=False)

    # Convert scenario_ids to a set for efficient comparison
    # scenario_ids_set = set(scenario_ids)

    # # Convert unique_scenario_ids to a set for efficient comparison
    # unique_scenario_ids_set = set(unique_scenario_ids)

    # # Find the scenario IDs that are in scenario_ids but not in unique_scenario_ids
    # missing_ids = list(scenario_ids_set - unique_scenario_ids_set)

    # # Create a DataFrame with the missing IDs
    # missing_ids_df = pd.DataFrame({'scenario_id': missing_ids})

    # # Specify the file path for the CSV output
    # output_file = "missing_ids.csv"

    # # Write the missing IDs to a CSV file
    # missing_ids_df.to_csv(output_file, index=False)

    # print(f"Missing scenario IDs have been written to '{output_file}'.")

    count_pos = 0
    count_neg = 0

    count_olADE = 0
    count_olFDE = 0
    count_both = 0

    sum_pos_ADE=0
    sum_neg_ADE=0
    sum_pos_FDE=0
    sum_neg_FDE=0

    avg_imp_ade = 0
    avg_imp_fde = 0

    # Open CSV files for writing
    with open('/home/vgrwbx/workspace/OL_trajectoryprediction/metrics_files/positive_05.csv', 'w', newline='') as csv_file_pos, \
         open('/home/vgrwbx/workspace/OL_trajectoryprediction/metrics_files/neg_05.csv', 'w', newline='') as csv_file_neg:

        writer_pos = csv.writer(csv_file_pos)
        writer_neg = csv.writer(csv_file_neg)

        # Iterate over each scenario ID
        for scenario_id in unique_scenario_ids:
            # Filter DataFrames for the current scenario ID
            filtered_df = df[df['scenario_id'] == scenario_id]
            filtered_dfol = df_ol[df_ol['scenario_id'] == scenario_id]

            improvADE, improvFDE = calculate_percentage_improvement_sum(filtered_dfol, filtered_df)

            if sum(filtered_dfol['val_minADE']) <= sum(filtered_df['val_minADE']):
                count_olADE+=1
                avg_imp_ade += improvADE
            if sum(filtered_dfol['val_minFDE']) <= sum(filtered_df['val_minFDE']):
                count_olFDE+=1
                avg_imp_fde+=improvFDE
            if (sum(filtered_dfol['val_minADE']) <= sum(filtered_df['val_minADE'])) and (sum(filtered_dfol['val_minFDE']) <= sum(filtered_df['val_minFDE'])):
                count_both+=1

            # Filter DataFrames for timesteps greater than or equal to 70
            filtered_df = filtered_df[filtered_df['timestep'] >= 80]
            filtered_dfol = filtered_dfol[filtered_dfol['timestep'] >= 80]

            # print('NOOL:',filtered_df)
            # print('OL:',filtered_dfol)

            # Reset index of both dataframes
            filtered_dfol_reset = filtered_dfol.reset_index(drop=True)
            filtered_df_reset = filtered_df.reset_index(drop=True)

            # if((filtered_dfol_reset['val_minADE'] < filtered_df_reset['val_minADE']).all()) or ((filtered_dfol_reset['val_minFDE'] < filtered_df_reset['val_minFDE']).all()):
            #     count_pos+=1
            # if((filtered_dfol_reset['val_minADE'] > filtered_df_reset['val_minADE']).all()) or ((filtered_dfol_reset['val_minFDE'] > filtered_df_reset['val_minFDE']).all()):
            #     count_neg+=1

            # Calculate percentage improvement
            percentage_improvement_ADE, percentage_improvement_FDE = calculate_percentage_improvement(filtered_dfol_reset, filtered_df_reset)

            # Check if both ADE and FDE differences are negative
            if (percentage_improvement_ADE < 0) and (percentage_improvement_FDE < 0):
                data = [scenario_id, percentage_improvement_ADE, percentage_improvement_FDE]
                write_to_file(writer_pos, data)
                count_pos += 1
                sum_pos_ADE+=percentage_improvement_ADE
                sum_pos_FDE+=percentage_improvement_FDE

            # Check if both ADE and FDE differences are positive
            if (percentage_improvement_ADE > 0) and (percentage_improvement_FDE > 0):
                data = [scenario_id, percentage_improvement_ADE, percentage_improvement_FDE]
                write_to_file(writer_neg, data)
                count_neg += 1
                sum_neg_ADE+=percentage_improvement_ADE
                sum_neg_FDE+=percentage_improvement_FDE

    # Print summary
    total_scenarios = len(unique_scenario_ids)
    print(f"{count_pos} improved with OL: {sum_pos_ADE/count_pos} {sum_pos_FDE/count_pos} as average improvement, {count_neg} have worsened {sum_neg_ADE/count_neg} {sum_neg_FDE/count_neg}, out of {total_scenarios}")
    print(f"{count_olADE} improved OL ADE {avg_imp_ade/count_olADE}, {count_olFDE} have improved OL FDE {avg_imp_fde/count_olFDE}, and {count_both} improved both metrics, out of {total_scenarios}")


if __name__ == "__main__":
    main()