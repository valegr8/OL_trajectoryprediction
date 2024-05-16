from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--log_id', type=str, default="7103fee0-bd5e-4fa3-a8e0-f9753ca1ecf7") # unique log identifier
    parser.add_argument('--filepath', type=str, default="/home/vgrwbx//workspace/OL_trajectoryprediction/val_metrics.csv")
    parser.add_argument('--noolfilepath', type=str, default="/home/vgrwbx//workspace/OL_trajectoryprediction/metrics_files/val_metrics_nool.csv")
    args = parser.parse_args()

    # Read the file into a pandas DataFrame
    df = pd.read_csv(args.filepath)
    dfnool = pd.read_csv(args.noolfilepath)

    # Convert val_minMR column to boolean
    # df['val_minMR'] = df['val_minMR'].astype(bool)

    # Filter the DataFrame based on the desired scenario_id
    filtered_df = df[df['scenario_id'] == args.log_id]
    print(filtered_df)
    filtered_dfnool = dfnool[dfnool['scenario_id'] == args.log_id]
    print(filtered_dfnool)
    # Add a column representing the iteration number
    filtered_df['iteration'] = filtered_df.groupby('scenario_id').cumcount() + 1
    filtered_dfnool['iteration'] = filtered_dfnool.groupby('scenario_id').cumcount() + 1

    # Plot the metrics over time
    plt.figure(figsize=(10, 6))
    # plt.plot(filtered_df['iteration'], filtered_df['val_Brier'], label='val_Brier')
    plt.plot(filtered_df['iteration'], filtered_df['val_minADE'], label='val_minADE_OL')
    plt.plot(filtered_dfnool['iteration'], filtered_dfnool['val_minADE'], label='val_minADE')
    # plt.plot(filtered_dfnool['iteration'], filtered_dfnool['val_minAHE'], label='val_minAHE')
    plt.plot(filtered_dfnool['iteration'], filtered_dfnool['val_minFDE'], label='val_minFDE')
    plt.plot(filtered_df['iteration'], filtered_df['val_minFDE'], label='val_minFDE_OL')
    # plt.plot(filtered_df['iteration'], filtered_df['val_minFHE'], label='val_minFHE')
    # plt.plot(filtered_df['iteration'], filtered_df['val_minMR'], label='val_minMR')

    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.title(f'Metrics Over Time for scenario_id: {args.log_id}')
    plt.legend()
    plt.grid(True)
    plt.show()
