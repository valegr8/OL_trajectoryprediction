import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def get_optimal_components(losses, method='BIC', min_components=1, max_components=10):
    """
    Get the optimal number of components for a Gaussian mixture model.
    
    Parameters:
        losses (array-like): Array of loss data.
        method (str): Method for determining the optimal number of components.
                      'BIC' for Bayesian Information Criterion, 'AIC' for Akaike Information Criterion.
        min_components (int): Minimum number of components to consider.
        max_components (int): Maximum number of components to consider.
    
    Returns:
        int: Optimal number of components.
    """
    bic_values = []
    aic_values = []

    n_components_range = range(min_components, max_components + 1)

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(losses)
        bic_values.append(gmm.bic(losses))
        aic_values.append(gmm.aic(losses))

    if method.upper() == 'BIC':
        return np.argmin(bic_values) + min_components
    elif method.upper() == 'AIC':
        return np.argmin(aic_values) + min_components
    else:
        raise ValueError("Invalid method. Choose either 'BIC' or 'AIC'.")

def fit_and_plot_gmm(losses, optimal_components):
    """
    Fit a Gaussian mixture model with the optimal number of components and plot the results.
    
    Parameters:
        losses (array-like): Array of loss data.
        optimal_components (int): Optimal number of components for the Gaussian mixture model.
    """
    # Fit Gaussian mixture model to the data with the optimal number of components
    gmm = GaussianMixture(n_components=optimal_components, random_state=42)
    gmm.fit(losses)

    # Generate data points to plot the fitted distribution
    x = np.linspace(min(losses), max(losses), 1000)
    pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))

    # Plot original loss data
    plt.hist(losses, bins=30, density=True, alpha=0.5, color='blue', label='Loss Data')

    # Plot fitted Gaussian mixture distribution
    plt.plot(x, pdf, color='red', linestyle='-', label='GMM Fit')

    plt.title('Fitted Gaussian Mixture Model')
    plt.xlabel('Losses')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Read the CSV file into a DataFrame
data = pd.read_csv("../losses.csv", header=None, names=["ID", "Index", "Loss"])

# Filter the loss dataset by the first ID and the first 10 timesteps
first_id_data = data[data["ID"] == data["ID"].iloc[0]][:90]

print(first_id_data)

# Extract the losses column
losses = first_id_data["Loss"].values

# Reshape losses to a column vector
losses = losses.reshape(-1, 1)

# Determine the optimal number of components using BIC or AIC
optimal_components = get_optimal_components(losses, method='BIC')

print('optimal components: ',optimal_components)

# Fit and plot Gaussian mixture model with the optimal number of components
fit_and_plot_gmm(losses, optimal_components)