# analyze and identify edge case scenarios

import pandas as pd
import matplotlib.pyplot as plt

def analyze_csv(filename):
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Convert the "val_minMR" column to boolean
    df["val_minMR"] = df["val_minMR"].astype(int)

    # Print some statistics
    print("Mean Brier Score:", df["val_Brier"].mean())
    print("Mean MinADE:", df["val_minADE"].mean())
    print("Mean MinAHE:", df["val_minAHE"].mean())
    print("Mean MinFDE:", df["val_minFDE"].mean())
    print("Mean MinFHE:", df["val_minFHE"].mean())
    print("Mean MinMR:", df["val_minMR"].mean())

    # # Plot the distribution of the "val_minMR" column
    # plt.figure(figsize=(6, 4))
    # plt.hist(df["val_minMR"], bins=2, edgecolor="black")
    # plt.title("Distribution of val_minMR")
    # plt.xlabel("val_minMR")
    # plt.ylabel("Frequency")
    # plt.xticks([0, 1], ["False", "True"])
    # plt.show()

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot the distribution of the "val_minADE" column as a histogram
    n_ade, bins_ade, _ = axes[0].hist(df["val_minADE"], bins=120, edgecolor="black", alpha=0.7, label="val_minADE")
    axes[0].set_title("Distribution of val_minADE")
    axes[0].set_xlabel("val_minADE")
    axes[0].set_ylabel("Frequency")

    # Plot the distribution of the "val_minFDE" column as a histogram
    n_fde, bins_fde, _ = axes[1].hist(df["val_minFDE"], bins=120, edgecolor="black", alpha=0.7, label="val_minFDE")
    axes[1].set_title("Distribution of val_minFDE")
    axes[1].set_xlabel("val_minFDE")
    axes[1].set_ylabel("Frequency")

    # Calculate cumulative distribution function
    cdf_ade = n_ade.cumsum() / n_ade.sum()
    cdf_fde = n_fde.cumsum() / n_fde.sum()

    # Find value after which 80% of the samples are located
    threshold_ade = bins_ade[1:][cdf_ade > 0.8][0]
    threshold_fde = bins_fde[1:][cdf_fde > 0.8][0]

    # Print the results
    print(f"For val_minADE: {threshold_ade}, 80% of samples are below this value.")
    print(f"For val_minFDE: {threshold_fde}, 80% of samples are below this value.")

    # Calculate the total number of samples
    total_samples = len(df)

    # Calculate the number of samples that are below the thresholds
    samples_below_ade = len(df[df["val_minADE"] < threshold_ade])
    samples_below_fde = len(df[df["val_minFDE"] < threshold_fde])

    # Print the number of samples that are below the thresholds
    print(f"Number of samples below the threshold for val_minADE: {samples_below_ade} out of {total_samples} ({samples_below_ade / total_samples * 100:.2f}%)")
    print(f"Number of samples below the threshold for val_minFDE: {samples_below_fde} out of {total_samples} ({samples_below_fde / total_samples * 100:.2f}%)")


    # Create a mask for samples above the threshold for val_minADE or val_minFDE
    mask = (df["val_minADE"] > threshold_ade) & (df["val_minFDE"] > threshold_fde)

    # Create a new DataFrame with samples above the threshold
    above_threshold_df = df[mask]

    # Remove duplicates based on the 'scenario_id' column
    above_threshold_df = above_threshold_df.drop_duplicates(subset="scenario_id")

    plt.tight_layout()
    plt.show()

    return above_threshold_df



above_threshold_df = analyze_csv("val_metrics.csv")




# Save the resulting DataFrame to a CSV file
# above_threshold_df.to_csv("metrics_files/above_threshold_samples.csv", index=False)

analyze_csv("metrics_files/above_threshold_samples.csv")

plt.show()