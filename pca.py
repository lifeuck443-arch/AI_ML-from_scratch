import numpy as np

def pca_from_scratch(X, n_components=None):
    """
    Perform PCA from scratch using the mathematical definition.
    
    Parameters
    ----------
    X : numpy.ndarray (n_samples, n_features)
        Input data
    n_components : int or None
        Number of principal components to keep. If None, return all.
    
    Returns
    -------
    X_transformed : numpy.ndarray
        Data projected onto principal component axes
    components : numpy.ndarray
        Principal component directions (eigenvectors)
    explained_variance : numpy.ndarray
        Eigenvalues corresponding to components
    """
    
    # 1. Center the data (subtract the mean)
    X_meaned = X - np.mean(X, axis=0)
    
    # 2. Compute the covariance matrix
    covariance_matrix = np.cov(X_meaned, rowvar=False)
    
    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 4. Sort eigenvalues/eigenvectors descending
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # 5. Select number of components
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]

    # 6. Project data onto components
    X_transformed = np.dot(X_meaned, eigenvectors)
    
    return X_transformed, eigenvectors, eigenvalues


# Example usage:
if __name__ == "__main__":
    # Example 2D dataset
    X = np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ])

    X_pca, components, var = pca_from_scratch(X, n_components=1)

    print("Projected Data:\n", X_pca)
    print("\nPrincipal Component (Eigenvector):\n", components)
    print("\nExplained Variance (Eigenvalue):\n", var[0])
