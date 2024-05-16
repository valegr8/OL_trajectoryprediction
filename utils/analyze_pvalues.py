from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--log_id', type=str, default="7103fee0-bd5e-4fa3-a8e0-f9753ca1ecf7") # unique log identifier
    parser.add_argument('--filepath', type=str, default="/home/vgrwbx//workspace/OL_trajectoryprediction/pvalues.csv")
    args = parser.parse_args()

    # Read the file into a pandas DataFrame
    df = pd.read_csv(args.filepath)

    # Filter the DataFrame based on the desired scenario_id
    filtered_df = df[df['scenario_id'] == args.log_id]

    # Plotting the data
    plt.figure(figsize=(10, 6))

    # Plot each scenario separately
    for scenario_id, group in filtered_df.groupby('scenario_id'):
        # Plot the p_values against timestep
        plt.plot(group['timestep'], group['p_value'], label=scenario_id)

    # Add red dashed lines at y=0.05 and y=0.95
    plt.axhline(y=0.05, color='red', linestyle='--', label='Threshold (0.05)')
    plt.axhline(y=0.95, color='red', linestyle='--', label='Threshold (0.95)')

    # Mark points where p_value is below 0.05 or above 0.95 with red dots
    plt.scatter(filtered_df[filtered_df['p_value'] < 0.05]['timestep'], filtered_df[filtered_df['p_value'] < 0.05]['p_value'], color='red', label='Suspitious Point, Below 0.05')
    plt.scatter(filtered_df[filtered_df['p_value'] > 0.95]['timestep'], filtered_df[filtered_df['p_value'] > 0.95]['p_value'], color='red', label='Suspitious Point, Above 0.95')

    # Set plot labels and legend
    plt.title('Scenario-wise p_value Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('p_value')
    plt.legend()
    plt.grid(True)
    plt.show()