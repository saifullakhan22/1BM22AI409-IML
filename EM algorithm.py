#6

import numpy as np
from scipy.stats import norm

# Function to generate synthetic data from two Gaussian distributions
def generate_data(n_samples):
    np.random.seed(42)
    data = np.concatenate([np.random.normal(0, 1, int(0.5 * n_samples)),
                           np.random.normal(5, 1, int(0.5 * n_samples))])
    np.random.shuffle(data)
    return data

# EM Algorithm for Gaussian Mixture Model
def em_algorithm(data, n_components, max_iterations=100, tolerance=1e-4):
    # Initialization
    weights = np.ones(n_components) / n_components
    means = np.linspace(data.min(), data.max(), n_components)
    variances = np.ones(n_components)

    for iteration in range(max_iterations):
        # Expectation step
        responsibilities = np.array([weights[i] * norm.pdf(data, means[i], np.sqrt(variances[i])) for i in range(n_components)])
        responsibilities /= responsibilities.sum(axis=0)

        # Maximization step
        weights = responsibilities.sum(axis=1) / len(data)
        means = (responsibilities * data).sum(axis=1) / responsibilities.sum(axis=1)
        variances = (responsibilities * (data - means[:, np.newaxis])**2).sum(axis=1) / responsibilities.sum(axis=1)

        # Check for convergence
        if np.abs(weights.sum() - 1) < tolerance and all(np.abs(weights - 1/n_components) < tolerance):
            break

    return weights, means, variances

if __name__ == "__main__":
    # Generate synthetic data
    data = generate_data(1000)

    # Number of components in the mixture
    n_components = 2

    # Run EM algorithm
    weights, means, variances = em_algorithm(data, n_components)

    # Display results
    print("Weights:", weights)
    print("Means:", means)
    print("Variances:", variances)
